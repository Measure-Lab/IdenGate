import os
import time
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import thop
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
import shutil


NPZ_PATH = "/home/ubuntu/dataset/MedMNIST/dermamnist_224.npz"
NUM_CLASSES = 7
N_CHANNELS = 3
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 150
AMP = True
OUT_DIR = "./outputs"
LOG_CSV = os.path.join(OUT_DIR, "training_log.csv")
BEST_PTH = os.path.join(OUT_DIR, "cmanet_blood_dp_best.pth")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def set_seed(seed=42):

    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

        self.img_size = img_size

        aug_list = [
            transforms.RandomCrop(img_size, padding=16),
            transforms.RandomHorizontalFlip(),
        ] if aug else []


        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            *aug_list,
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.dim() == 3 and t.shape[0] == 1 else t),
            transforms.Normalize([0.5]*3, [0.5]*3),
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

def load_blood_npz(npz_path=NPZ_PATH):
    data = np.load(npz_path)
    x_tr, y_tr = data["train_images"], data["train_labels"]
    x_va, y_va = data["val_images"],   data["val_labels"]
    x_te, y_te = data["test_images"],  data["test_labels"]
    x_tr = np.concatenate([x_tr, x_va], axis=0)
    y_tr = np.concatenate([y_tr, y_va], axis=0)
    return (x_tr, y_tr), (x_te, y_te)


(x_train, y_train), (x_test, y_test) = load_blood_npz()
trainset = BloodNPZDataset(x_train, y_train, aug=True)
testset  = BloodNPZDataset(x_test,  y_test,  aug=False)


NUM_WORKERS = 8
train_loader = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0), prefetch_factor=2
)
test_loader  = DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0), prefetch_factor=2
)


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
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.view(b, c, h*w).permute(2, 0, 1)
        if not self.use_conditioned:
            out = self.encoder(seq)
            return out.permute(1, 2, 0).view(b, c, h, w)
        pooled = x.mean(dim=(2,3), keepdim=True)
        phi_q = 1.0 + self.alpha*torch.tanh(self.gq(pooled)).view(b, c)
        phi_k = 1.0 + self.alpha*torch.tanh(self.gk(pooled)).view(b, c)
        phi_v = 1.0 + self.alpha*torch.tanh(self.gv(pooled)).view(b, c)
        phi_q, phi_k, phi_v = phi_q.unsqueeze(0), phi_k.unsqueeze(0), phi_v.unsqueeze(0)
        enc = self.encoder
        q, k, v = seq*phi_q, seq*phi_k, seq*phi_v
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

def compute_model_metrics_single(model):
    sample = torch.randn(1, N_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)
    try:
        flops, params = thop.profile(model, inputs=(sample,), verbose=False)
        print(f" Model FLOPs: {flops/1e9:.2f} G, Params: {params/1e6:.2f} M")
    except Exception as e:
        print(f"[WARN] thop fail：{e}")

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, use_amp: bool):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        pred = out.argmax(1)
        running_correct += (pred == y).sum().item()
        total += y.size(0)
    avg_loss = running_loss / max(1, len(loader))
    acc = 100.0 * running_correct / max(1, total)
    return avg_loss, acc

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool):

    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0


    cls_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    cls_total   = np.zeros(NUM_CLASSES, dtype=np.int64)


    all_probs = []
    all_y = []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
                cls_total[cls]   += mask.sum().item()
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
            per_class_acc.append(float('nan'))


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
                roc_auc_score(all_y, y_score)
            ]
        else:
            auc_macro = roc_auc_score(all_y, all_probs, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(all_y, all_probs, multi_class='ovr', average='weighted')
            auc_per_class = roc_auc_score(all_y, all_probs, multi_class='ovr', average=None).tolist()
    except Exception as e:
        print(f"[WARN] AUC fail：{e}")
        auc_macro, auc_weighted = float('nan'), float('nan')
        auc_per_class = [float('nan')] * all_probs.shape[1]


    return avg_loss, acc, per_class_acc, auc_macro, auc_weighted, auc_per_class

def build_csv_header():
    header = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr", "epoch_time_sec"]
    header += [f"test_acc_c{i}" for i in range(NUM_CLASSES)]
    header += ["test_auc_macro", "test_auc_weighted"]
    header += [f"test_auc_c{i}" for i in range(NUM_CLASSES)]
    return header

def write_csv_header(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(build_csv_header())

def append_csv_row(path, row):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        script_src = os.path.abspath(__file__)
        base = os.path.basename(script_src)
        snap_name = f"{os.path.splitext(base)[0]}_{time.strftime('%Y%m%d-%H%M%S')}.py"

        dst_latest = os.path.join(OUT_DIR, base)
        dst_snap   = os.path.join(OUT_DIR, snap_name)

        shutil.copyfile(script_src, dst_latest)
        shutil.copyfile(script_src, dst_snap)
        print(f"[INFO] Script copied to:\n  - {dst_latest}\n  - {dst_snap}")
    except Exception as e:
        print(f"[WARN] Failed to copy script: {e}")

    torch.cuda.set_device(0)
    model = CMANet().to(DEVICE)

    compute_model_metrics_single(model)

    not_cuda = []
    for name, p in model.named_parameters():
        if p.device.type != "cuda":
            not_cuda.append(("param", name, p.device))
    for name, b in model.named_buffers():
        if b.device.type != "cuda":
            not_cuda.append(("buffer", name, b.device))
    if not_cuda:
        print("[WARN] Found tensors not on CUDA before DataParallel:")
        for kind, name, dev in not_cuda:
            print(f"  - {kind}: {name} @ {dev}")
        model.to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device='cuda', enabled=AMP)

    write_csv_header(LOG_CSV)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, DEVICE, AMP
        )
        scheduler.step()

        test_loss, test_acc, per_cls_acc, auc_macro, auc_weighted, auc_per_class = evaluate(
            model, test_loader, criterion, DEVICE, AMP
        )

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(f"Epoch {epoch:03d} | Train {train_loss:.4f}/{train_acc:.2f}% | "
              f"Test {test_loss:.4f}/{test_acc:.2f}% | "
              f"AUC(macro) {auc_macro:.4f} | LR {lr_now:.6f} | {dt:.1f}s")

        row = [
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.3f}",
            f"{test_loss:.6f}",
            f"{test_acc:.3f}",
            f"{lr_now:.8f}",
            f"{dt:.3f}",
        ]
        row += [f"{(a if np.isfinite(a) else float('nan')):.3f}" for a in per_cls_acc]
        row += [f"{(auc_macro if np.isfinite(auc_macro) else float('nan')):.6f}",
                f"{(auc_weighted if np.isfinite(auc_weighted) else float('nan')):.6f}"]
        row += [f"{(a if np.isfinite(a) else float('nan')):.6f}" for a in auc_per_class]

        append_csv_row(LOG_CSV, row)

        if test_acc >= best_acc:
            best_acc = test_acc
            target = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(target.state_dict(), BEST_PTH)

    print(f"Training done. Best Test Acc: {best_acc:.2f}%")
    print(f"Best weights saved to: {BEST_PTH}")
    print(f"CSV log saved to: {LOG_CSV}")
