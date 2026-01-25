import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# get csv

NPZ_PATH = "/home/ubuntu/dataset/MedMNIST/dermamnist_224.npz"
WEIGHT_PTH = "/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/42/cmanet_blood_dp_best.pth"

NUM_CLASSES = 7
N_CHANNELS = 3
IMG_SIZE = 224

BATCH_SIZE = 128
NUM_WORKERS = 8

AMP = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

OUT_DIR = "/home/ubuntu/PycharmProjects/MIA/middle"
OUT_CSV = os.path.join(OUT_DIR, "sensor_noise_robustness.csv")


NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25]


def set_seed(seed=42):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

set_seed(42)


class DermNPZSensorNoiseDataset(Dataset):

    def __init__(self, images, labels, noise_std: float = 0.0, img_size=IMG_SIZE):
        self.images = images

        lbl = np.array(labels)
        if lbl.ndim == 2:
            if lbl.shape[1] == 1:
                lbl = lbl.squeeze(1)
            else:
                lbl = lbl.argmax(1)
        elif lbl.ndim != 1:
            lbl = lbl.reshape(-1)
        self.labels = lbl.astype(np.int64)

        self.noise_std = float(noise_std)
        self.img_size = img_size

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5]*3, [0.5]*3)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]

        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        img = np.ascontiguousarray(img).astype(np.uint8)
        pil = self.to_pil(img)

        if pil.size != (self.img_size, self.img_size):
            pil = pil.resize((self.img_size, self.img_size))

        x = self.to_tensor(pil)

        if x.dim() == 3 and x.shape[0] == 1:
            x = x.repeat(3, 1, 1)

        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = torch.clamp(x + noise, 0.0, 1.0)

        x = self.normalize(x)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y

def load_npz(npz_path=NPZ_PATH):
    data = np.load(npz_path)
    x_te, y_te = data["test_images"], data["test_labels"]
    return x_te, y_te


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
            d_model=dim, nhead=nhead, dim_feedforward=dim*mlp_ratio, dropout=dropout
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
        seq = x.view(b, c, h*w).permute(2, 0, 1)

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
def eval_acc_auc(model: nn.Module, loader: DataLoader, use_amp: bool = True):
    model.eval()
    correct, total = 0, 0
    all_probs, all_y = [], []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(use_amp and DEVICE.type == "cuda")):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

        pred = probs.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    acc = 100.0 * correct / max(1, total)

    all_probs = np.concatenate(all_probs, axis=0)
    all_y = np.concatenate(all_y, axis=0).reshape(-1)


    try:
        if all_probs.shape[1] == 2:
            auc_macro = roc_auc_score(all_y, all_probs[:, 1])
            auc_weighted = auc_macro
        else:
            auc_macro = roc_auc_score(all_y, all_probs, multi_class="ovr", average="macro")
            auc_weighted = roc_auc_score(all_y, all_probs, multi_class="ovr", average="weighted")
    except Exception as e:
        print(f"[WARN] AUC failed: {e}")
        auc_macro, auc_weighted = float("nan"), float("nan")

    return acc, auc_macro, auc_weighted

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise_intensity", "acc", "auc_macro", "auc_weighted"])
        w.writerows(rows)

def main():

    x_te, y_te = load_npz(NPZ_PATH)
    print(f"[INFO] Test set: images={x_te.shape}, labels={np.array(y_te).shape}")

    model = CMANet(num_classes=NUM_CLASSES, n_channels=3).to(DEVICE)

    ckpt = torch.load(WEIGHT_PTH, map_location="cpu")

    state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt

    new_state = {}
    for k, v in state.items():
        k2 = k[len("module."):] if k.startswith("module.") else k

        if "total_ops" in k2 or "total_params" in k2:
            continue

        new_state[k2] = v

    model.load_state_dict(new_state, strict=True)
    model.eval()

    print(f"[OK] Loaded weights (strict=True): {WEIGHT_PTH}")

    results = []
    for sigma in NOISE_LEVELS:
        ds = DermNPZSensorNoiseDataset(x_te, y_te, noise_std=sigma, img_size=IMG_SIZE)
        loader = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0), prefetch_factor=2
        )

        acc, auc_macro, auc_weighted = eval_acc_auc(model, loader, use_amp=AMP)
        print(f"[Noise std={sigma:.3f}]  Acc={acc:.2f}%  AUC(macro)={auc_macro:.4f}  AUC(weighted)={auc_weighted:.4f}")
        results.append([f"{sigma:.3f}", f"{acc:.4f}", f"{auc_macro:.6f}", f"{auc_weighted:.6f}"])

    write_csv(OUT_CSV, results)
    print(f"[DONE] Saved CSV -> {OUT_CSV}")

if __name__ == "__main__":
    main()
