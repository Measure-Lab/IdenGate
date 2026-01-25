import os
import csv
import random
import numpy as np
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from contextlib import nullcontext


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False
SEED = 42
set_seed(SEED)

WEIGHTS   = "/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/42/cmanet_blood_dp_best.pth"

NPZ_PATH  = "/home/ubuntu/dataset/MedMNIST/dermamnist_224.npz"

NUM_CLASSES = 7
N_CHANNELS  = 3
IMG_SIZE    = 224
BATCH_SIZE  = 128
NUM_WORKERS = 8
AMP         = True

RUN_TAG = f"retina_alme_seed{SEED}"

_dl_generator = torch.Generator()
_dl_generator.manual_seed(42)

def _seed_worker(worker_id: int):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MedMNISTNPZDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, aug: bool = False, img_size: int = 224):
        self.images = images

        lbl = np.array(labels)
        if lbl.ndim == 2:
            if lbl.shape[1] == 1:
                lbl = lbl.squeeze(1)
            else:
                lbl = lbl.argmax(1)
        elif lbl.ndim == 1:
            pass
        else:
            lbl = lbl.reshape(-1)
        self.labels = lbl.astype(np.int64)

        aug_list = [
            transforms.RandomCrop(img_size, padding=16),
            transforms.RandomHorizontalFlip(),
        ] if aug else []

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            *aug_list,
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.dim() == 3 and t.shape[0] == 1 else t),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i: int):
        img = self.images[i]

        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        img = np.ascontiguousarray(img)
        x = self.tf(img.astype(np.uint8) if img.dtype != np.uint8 else img)
        y = int(self.labels[i])
        return x, y, i

def load_npz_train_test(npz_path: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    data = np.load(npz_path)
    x_tr, y_tr = data["train_images"], data["train_labels"]
    x_va, y_va = data["val_images"],   data["val_labels"]
    x_te, y_te = data["test_images"],  data["test_labels"]
    x_tr = np.concatenate([x_tr, x_va], axis=0)
    y_tr = np.concatenate([y_tr, y_va], axis=0)
    return (x_tr, y_tr), (x_te, y_te)

class ALME(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        z = self.dwconv(x)
        z = self.bn(z)
        z = z * self.se(z)
        z = self.proj(z)
        return z

class ConditionedSelfAttention(nn.Module):
    def __init__(self, dim, nhead=4, mlp_ratio=4, dropout=0.1, use_conditioned=False, alpha=0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * mlp_ratio, dropout=dropout
        )
        self.use_conditioned = use_conditioned
        self.alpha = alpha
        self.gq = nn.Conv2d(dim, dim, 1, groups=dim, bias=True)
        self.gk = nn.Conv2d(dim, dim, 1, groups=dim, bias=True)
        self.gv = nn.Conv2d(dim, dim, 1, groups=dim, bias=True)
        for m in [self.gq, self.gk, self.gv]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.view(b, c, h * w).permute(2, 0, 1)

        if not self.use_conditioned:
            out = self.encoder(seq)
            return out.permute(1, 2, 0).view(b, c, h, w)

        pooled = x.mean(dim=(2, 3), keepdim=True)
        phi_q = 1.0 + self.alpha * torch.tanh(self.gq(pooled)).view(b, c)
        phi_k = 1.0 + self.alpha * torch.tanh(self.gk(pooled)).view(b, c)
        phi_v = 1.0 + self.alpha * torch.tanh(self.gv(pooled)).view(b, c)
        phi_q, phi_k, phi_v = phi_q.unsqueeze(0), phi_k.unsqueeze(0), phi_v.unsqueeze(0)

        enc = self.encoder
        q, k, v = seq * phi_q, seq * phi_k, seq * phi_v
        attn_out, _ = enc.self_attn(q, k, v, need_weights=False)

        src = seq + enc.dropout1(attn_out)
        src = enc.norm1(src)
        ffn_out = enc.linear2(enc.dropout(enc.activation(enc.linear1(src))))
        src = src + enc.dropout2(ffn_out)
        src = enc.norm2(src)
        return src.permute(1, 2, 0).view(b, c, h, w)

class CMABlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.alme = ALME(dim, reduction)
        self.attn = ConditionedSelfAttention(dim)

    def forward(self, x):
        return x + self.attn(self.alme(x))

class CMANet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, n_channels=N_CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            CMABlock(64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            CMABlock(128),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            CMABlock(256),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

@torch.no_grad()
def export_preds_for_risk_coverage():

    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()

    out_dir = script_dir / "middle"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{RUN_TAG}_preds_for_rc.csv"
    out_npz = out_dir / f"{RUN_TAG}_preds_for_rc.npz"
    out_txt = out_dir / f"{RUN_TAG}_preds_for_rc.txt"

    (_, _), (x_te, y_te) = load_npz_train_test(NPZ_PATH)
    testset = MedMNISTNPZDataset(x_te, y_te, aug=False, img_size=IMG_SIZE)

    test_loader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=_seed_worker,
        generator=_dl_generator
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CMANet(num_classes=NUM_CLASSES, n_channels=N_CHANNELS).to(device)

    ckpt = torch.load(WEIGHTS, map_location=device)
    state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
    new_state = {}
    for k, v in state.items():
        k2 = k[len("module."):] if k.startswith("module.") else k
        if "total_ops" in k2 or "total_params" in k2:
            continue
        new_state[k2] = v

    model.load_state_dict(new_state, strict=True)
    model.eval()


    use_amp = AMP and (device.type == "cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    all_ids, all_y, all_probs = [], [], []
    for x, y, idx in tqdm(test_loader, desc="Export(Risk-Coverage preds)"):
        x = x.to(device, non_blocking=True)
        with autocast_ctx():
            logits = model(x)
            probs = F.softmax(logits, dim=1)

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(np.asarray(y))
        all_ids.append(np.asarray(idx))

    probs = np.concatenate(all_probs, axis=0).astype(np.float32)
    y_true = np.concatenate(all_y, axis=0).reshape(-1).astype(np.int64)
    sample_id = np.concatenate(all_ids, axis=0).reshape(-1).astype(np.int64)

    if y_true.ndim == 2:
        if y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)
        elif y_true.shape[1] == probs.shape[1]:
            y_true = y_true.argmax(axis=1)
        else:
            y_true = y_true.reshape(-1)

    y_pred = probs.argmax(axis=1).astype(np.int64)
    conf = probs.max(axis=1).astype(np.float32)
    correct = (y_pred == y_true).astype(np.int32)

    acc = float(correct.mean() * 100.0)
    n = int(len(y_true))

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["sample_id", "y_true", "y_pred", "conf"] + [f"prob_c{i}" for i in range(NUM_CLASSES)]
        w.writerow(header)
        for i in range(n):
            w.writerow([
                int(sample_id[i]),
                int(y_true[i]),
                int(y_pred[i]),
                f"{float(conf[i]):.8f}",
                *[f"{float(p):.8f}" for p in probs[i].tolist()],
            ])

    np.savez_compressed(
        out_npz,
        sample_id=sample_id,
        y_true=y_true,
        y_pred=y_pred,
        conf=conf,
        probs=probs,
        correct=correct,
        num_classes=np.array([NUM_CLASSES], dtype=np.int64),
    )

    lines = [
        f"[INFO] RUN_TAG: {RUN_TAG}",
        f"[INFO] N = {n}",
        f"[INFO] Test Top-1 Accuracy: {acc:.2f}%",
        "[OK] Risk–Coverage ready outputs saved:",
        f"  CSV: {out_csv}",
        f"  NPZ: {out_npz}",
        "[FIELDS] sample_id, y_true, y_pred, conf(=max softmax prob), probs, correct",
        "",
        "Next step (RC curve): sort by conf desc; sweep coverage; risk = 1 - accuracy on kept samples.",
    ]
    print("\n".join(lines))
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    export_preds_for_risk_coverage()







