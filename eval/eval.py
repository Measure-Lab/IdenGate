import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast
from sklearn.metrics import roc_auc_score


NUM_CLASSES = 5
N_CHANNELS = 3
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 8

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, "retinamnist_224.npz")
WEIGHT_PTH = os.path.join(SCRIPT_DIR, "cmanet_blood_dp_best.pth")
OUT_CSV = os.path.join(SCRIPT_DIR, "eval_results.csv")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

AMP = (DEVICE.type == "cuda")

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def set_seed(seed=42):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


set_seed(42)


class BloodNPZDataset(Dataset):
    def __init__(self, images, labels, aug=False, img_size=IMG_SIZE):
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

    def __getitem__(self, i):
        img = self.images[i]

        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        img = np.ascontiguousarray(img).astype(np.uint8)
        x = self.tf(img)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y


def load_retina_npz_test(npz_path):
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
    def __init__(self, dim, nhead=4, mlp_ratio=4, dropout=0.1, use_conditioned=True, alpha=0.1):
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
def evaluate(model, loader, criterion, device, use_amp: bool):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0

    cls_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    cls_total = np.zeros(NUM_CLASSES, dtype=np.int64)

    all_probs = []
    all_y = []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type='cuda', enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
            probs = torch.softmax(out, dim=1)

        running_loss += loss.item()

        pred = out.argmax(1)
        running_correct += (pred == y).sum().item()
        total += y.size(0)

        for cls in range(NUM_CLASSES):
            mask = (y == cls)
            if mask.any():
                cls_total[cls] += mask.sum().item()
                cls_correct[cls] += (pred[mask] == y[mask]).sum().item()

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    avg_loss = running_loss / max(1, len(loader))
    acc = 100.0 * running_correct / max(1, total)

    per_class_acc = []
    for cls in range(NUM_CLASSES):
        if cls_total[cls] > 0:
            per_class_acc.append(100.0 * cls_correct[cls] / cls_total[cls])
        else:
            per_class_acc.append(float("nan"))

    all_probs = np.concatenate(all_probs, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    if all_y.ndim == 2:
        if all_y.shape[1] == all_probs.shape[1]:
            all_y = all_y.argmax(axis=1)
        else:
            all_y = all_y.squeeze()

    try:
        C = all_probs.shape[1]
        if C == 2:
            y_score = all_probs[:, 1]
            auc_macro = roc_auc_score(all_y, y_score)
            auc_weighted = auc_macro
            auc_per_class = [
                roc_auc_score(all_y, 1.0 - y_score),
                roc_auc_score(all_y, y_score),
            ]
        else:
            auc_macro = roc_auc_score(all_y, all_probs, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(all_y, all_probs, multi_class='ovr', average='weighted')
            auc_per_class = roc_auc_score(all_y, all_probs, multi_class='ovr', average=None).tolist()
    except Exception as e:
        print(f"[WARN] AUC fail：{e}")
        auc_macro, auc_weighted = float("nan"), float("nan")
        auc_per_class = [float("nan")] * all_probs.shape[1]

    return avg_loss, acc, per_class_acc, auc_macro, auc_weighted, auc_per_class


def save_csv(path, test_loss, test_acc, per_cls_acc, auc_macro, auc_weighted, auc_per_class):
    header = ["test_loss", "test_acc"] \
             + [f"test_acc_c{i}" for i in range(NUM_CLASSES)] \
             + ["test_auc_macro", "test_auc_weighted"] \
             + [f"test_auc_c{i}" for i in range(NUM_CLASSES)]
    row = [f"{test_loss:.6f}", f"{test_acc:.3f}"] \
          + [f"{a:.3f}" if np.isfinite(a) else "nan" for a in per_cls_acc] \
          + [f"{auc_macro:.6f}" if np.isfinite(auc_macro) else "nan",
             f"{auc_weighted:.6f}" if np.isfinite(auc_weighted) else "nan"] \
          + [f"{a:.6f}" if np.isfinite(a) else "nan" for a in auc_per_class]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)


def load_weights_safely(model, weight_path, device):
    ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        if len(keys) > 0 and keys[0].startswith("module."):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print("[WARN] Missing keys (first 30):")
        for k in missing[:30]:
            print("  ", k)
    if unexpected:
        print("[WARN] Unexpected keys (first 30):")
        for k in unexpected[:30]:
            print("  ", k)


if __name__ == "__main__":
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"NPZ not found: {NPZ_PATH}")
    if not os.path.exists(WEIGHT_PTH):
        raise FileNotFoundError(f"Weight not found: {WEIGHT_PTH}")

    x_test, y_test = load_retina_npz_test(NPZ_PATH)
    testset = BloodNPZDataset(x_test, y_test, aug=False)
    test_loader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0), prefetch_factor=2
    )

    model = CMANet().to(DEVICE)
    load_weights_safely(model, WEIGHT_PTH, DEVICE)

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, per_cls_acc, auc_macro, auc_weighted, auc_per_class = evaluate(
        model, test_loader, criterion, DEVICE, AMP
    )

    print("\n========== EVAL RESULT ==========")
    print(f"NPZ:    {NPZ_PATH}")
    print(f"WEIGHT: {WEIGHT_PTH}")
    print(f"Device: {DEVICE} | AMP: {AMP}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Acc : {test_acc:.3f}%")
    print("Per-class Acc (%):")
    for i, a in enumerate(per_cls_acc):
        print(f"  c{i}: {a:.3f}" if np.isfinite(a) else f"  c{i}: nan")
    print(f"AUC Macro   : {auc_macro:.6f}" if np.isfinite(auc_macro) else "AUC Macro   : nan")
    print(f"AUC Weighted: {auc_weighted:.6f}" if np.isfinite(auc_weighted) else "AUC Weighted: nan")
    print("Per-class AUC:")
    for i, a in enumerate(auc_per_class):
        print(f"  c{i}: {a:.6f}" if np.isfinite(a) else f"  c{i}: nan")

    save_csv(OUT_CSV, test_loss, test_acc, per_cls_acc, auc_macro, auc_weighted, auc_per_class)
    print(f"\n[OK] CSV saved to: {OUT_CSV}")
