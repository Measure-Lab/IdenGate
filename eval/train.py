"""Train IDENGATE on MedMNIST-format NPZ datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import (
    CMANet,
    PAPER_NO_MGF_PARAMETER_COUNT,
    PAPER_PRIMARY_PARAMETER_COUNT,
    count_trainable_parameters,
    validate_reported_parameter_counts,
)

PAPER_SEEDS = (42, 43, 44, 45, 46)
VALID_CONTROLS = ("full", "no_mgf", "shuffle_mgf")


@dataclass(frozen=True)
class TrainConfig:
    npz_path: str
    output_dir: str
    control: str = "full"
    num_classes: int = 5
    in_channels: int = 3
    image_size: int = 224
    stem_stride: int = 4
    depth: int = 3
    base_dim: int = 64
    reduction: int = 16
    nhead: int = 4
    mlp_ratio: int = 2
    dropout: float = 0.1
    alpha: float = 0.1
    max_tokens: int = 4096
    epochs: int = 60
    batch_size: int = 128
    num_workers: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler_t0: int = 10
    scheduler_tmult: int = 2
    crop_padding: int = 16
    horizontal_flip: bool = True
    normalization_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalization_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    use_amp: bool = True
    deterministic: bool = True
    allow_tf32: bool = True
    data_parallel: bool = False


class MedMNISTNPZDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset wrapper for official MedMNIST NPZ arrays."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        image_size: int,
        augment: bool,
        crop_padding: int,
        horizontal_flip: bool,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.images = np.asarray(images)
        self.labels = normalize_labels(labels)
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("images and labels have different lengths")

        operations: list[Any] = [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
        ]
        if augment:
            operations.append(transforms.RandomCrop(image_size, padding=crop_padding))
            if horizontal_flip:
                operations.append(transforms.RandomHorizontalFlip())
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(ensure_three_channels),
                transforms.Normalize(list(mean), list(std)),
            ]
        )
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        image = np.ascontiguousarray(image)
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        x = self.transform(image)
        y = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return x, y


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2:
        if labels.shape[1] == 1:
            labels = labels[:, 0]
        else:
            labels = labels.argmax(axis=1)
    elif labels.ndim != 1:
        labels = labels.reshape(-1)
    return labels.astype(np.int64, copy=False)


def ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        return tensor.repeat(3, 1, 1)
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected one or three channels, got {tensor.shape[0]}")
    return tensor


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def load_official_npz_splits(
    npz_path: str | Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = Path(npz_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    required = {
        "train_images",
        "train_labels",
        "val_images",
        "val_labels",
        "test_images",
        "test_labels",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"NPZ file is missing arrays: {missing}")
        splits = {
            "train": (data["train_images"].copy(), data["train_labels"].copy()),
            "val": (data["val_images"].copy(), data["val_labels"].copy()),
            "test": (data["test_images"].copy(), data["test_labels"].copy()),
        }

    for name, (images, labels) in splits.items():
        normalized = normalize_labels(labels)
        if len(images) != len(normalized):
            raise ValueError(f"{name} images/labels length mismatch")
    return splits


def set_seed(seed: int, *, deterministic: bool, allow_tf32: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(
    config: TrainConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    splits = load_official_npz_splits(config.npz_path)
    train_dataset = MedMNISTNPZDataset(
        *splits["train"],
        image_size=config.image_size,
        augment=True,
        crop_padding=config.crop_padding,
        horizontal_flip=config.horizontal_flip,
        mean=config.normalization_mean,
        std=config.normalization_std,
    )
    val_dataset = MedMNISTNPZDataset(
        *splits["val"],
        image_size=config.image_size,
        augment=False,
        crop_padding=config.crop_padding,
        horizontal_flip=False,
        mean=config.normalization_mean,
        std=config.normalization_std,
    )
    test_dataset = MedMNISTNPZDataset(
        *splits["test"],
        image_size=config.image_size,
        augment=False,
        crop_padding=config.crop_padding,
        horizontal_flip=False,
        mean=config.normalization_mean,
        std=config.normalization_std,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    common: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    if config.num_workers > 0:
        common["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }
    return train_loader, val_loader, test_loader, sizes


def build_model(config: TrainConfig) -> CMANet:
    if config.control not in VALID_CONTROLS:
        raise ValueError(f"control must be one of {VALID_CONTROLS}")

    model = CMANet(
        num_classes=config.num_classes,
        in_channels=config.in_channels,
        depth=config.depth,
        base_dim=config.base_dim,
        reduction=config.reduction,
        nhead=config.nhead,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        alpha=config.alpha,
        use_mgf=config.control != "no_mgf",
        stem_stride=config.stem_stride,
        max_tokens=config.max_tokens,
    )

    parameter_count = count_trainable_parameters(model)
    if (
        config.depth == 3
        and config.base_dim == 64
        and config.num_classes == 5
        and config.in_channels == 3
    ):
        expected = (
            PAPER_NO_MGF_PARAMETER_COUNT
            if config.control == "no_mgf"
            else PAPER_PRIMARY_PARAMETER_COUNT
        )
        if parameter_count != expected:
            raise RuntimeError(
                f"Parameter-count mismatch: observed {parameter_count:,}, expected {expected:,}"
            )
    return model


def unwrap_model(model: nn.Module) -> CMANet:
    module = model.module if isinstance(model, nn.DataParallel) else model
    if not isinstance(module, CMANet):
        raise TypeError(f"Unexpected model type: {type(module)}")
    return module


def forward_for_control(model: nn.Module, x: torch.Tensor, control: str) -> torch.Tensor:
    return model(x, shuffle_gates=(control == "shuffle_mgf"))


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities)
    try:
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(labels, probabilities[:, 1]))
        return float(
            roc_auc_score(
                labels,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def classwise_ovr_auc(labels: np.ndarray, probabilities: np.ndarray) -> list[float]:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities)
    values: list[float] = []
    for class_index in range(probabilities.shape[1]):
        binary = (labels == class_index).astype(np.int64)
        try:
            values.append(float(roc_auc_score(binary, probabilities[:, class_index])))
        except ValueError:
            values.append(float("nan"))
    return values


def amp_context(device: torch.device, enabled: bool):
    return autocast(device_type=device.type, enabled=enabled)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp_enabled: bool,
    control: str,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, amp_enabled):
            logits = forward_for_control(model, images, control)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch = labels.shape[0]
        loss_sum += float(loss.detach().item()) * batch
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += batch

    return {
        "loss": loss_sum / max(count, 1),
        "accuracy_pct": 100.0 * correct / max(count, 1),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp_enabled: bool,
    control: str,
    description: str,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    count = 0
    logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []

    for images, labels in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_context(device, amp_enabled):
            logits = forward_for_control(model, images, control)
            loss = criterion(logits, labels)
        batch = labels.shape[0]
        loss_sum += float(loss.item()) * batch
        count += batch
        logits_parts.append(logits.float().cpu())
        labels_parts.append(labels.cpu())

    logits = torch.cat(logits_parts, dim=0)
    labels = torch.cat(labels_parts, dim=0)
    probabilities = torch.softmax(logits, dim=1).numpy()
    labels_np = labels.numpy()
    predictions = probabilities.argmax(axis=1)
    accuracy = 100.0 * float(np.mean(predictions == labels_np))
    auc = macro_auc(labels_np, probabilities)

    return {
        "loss": loss_sum / max(count, 1),
        "accuracy_pct": accuracy,
        "macro_auc": auc,
        "macro_auc_pct": 100.0 * auc if np.isfinite(auc) else float("nan"),
        "classwise_auc": classwise_ovr_auc(labels_np, probabilities),
        "logits": logits,
        "labels": labels,
        "probabilities": probabilities,
        "predictions": predictions,
    }


def json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if key in {"logits", "labels", "probabilities", "predictions"}:
            continue
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def write_history_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                "epoch",
                "train_loss",
                "train_accuracy_pct",
                "val_loss",
                "val_accuracy_pct",
                "val_macro_auc_pct",
                "learning_rate",
                "epoch_time_sec",
                "selected",
            ]
        )


def append_history(path: Path, row: list[Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    seed: int,
    config: TrainConfig,
    val_metrics: dict[str, Any],
    dataset_sha256: str,
) -> None:
    module = unwrap_model(model)
    payload = {
        "format_version": 1,
        "model_name": "IDENGATE",
        "model_state_dict": module.state_dict(),
        "model_config": module.model_config(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "seed": int(seed),
        "control": config.control,
        "train_config": asdict(config),
        "validation_metrics": json_safe_metrics(val_metrics),
        "dataset_sha256": dataset_sha256,
        "git_commit": git_commit(),
        "created_unix_time": time.time(),
    }
    torch.save(payload, path)


def load_checkpoint_model(path: Path, device: torch.device) -> tuple[CMANet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint lacks model_config/model_state_dict")
    model = CMANet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    return model, checkpoint


def t_interval_95(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return {"mean": mean, "sd": 0.0, "ci95_low": mean, "ci95_high": mean}
    sd = float(arr.std(ddof=1))
    # Manuscript five-seed analyses use Student's t with 4 df.
    critical = 2.7764451051977987 if arr.size == 5 else 1.959963984540054
    half = critical * sd / math.sqrt(arr.size)
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def environment_manifest(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    }


def run_seed(config: TrainConfig, seed: int, device: torch.device) -> dict[str, Any]:
    set_seed(seed, deterministic=config.deterministic, allow_tf32=config.allow_tf32)
    train_loader, val_loader, test_loader, split_sizes = build_loaders(config, seed)
    dataset_hash = sha256_file(config.npz_path)

    seed_dir = Path(config.output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    history_path = seed_dir / "history.csv"
    best_path = seed_dir / "best_validation_macro_auc.pt"
    last_path = seed_dir / "last.pt"
    write_history_header(history_path)

    model = build_model(config).to(device)
    geometry = model.token_geometry(config.image_size)
    print(
        f"seed={seed} control={config.control} params={count_trainable_parameters(model):,} "
        f"token_grids={geometry}"
    )

    if config.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.scheduler_t0,
        T_mult=config.scheduler_tmult,
    )
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=amp_enabled)

    best_auc = -float("inf")
    best_epoch = -1
    best_val_metrics: dict[str, Any] | None = None

    for epoch in range(1, config.epochs + 1):
        start = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            amp_enabled=amp_enabled,
            control=config.control,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
            control=config.control,
            description="validation",
        )
        scheduler.step()
        elapsed = time.time() - start
        val_auc = float(val_metrics["macro_auc"])
        selected = bool(np.isfinite(val_auc) and val_auc > best_auc)

        if selected:
            best_auc = val_auc
            best_epoch = epoch
            best_val_metrics = json_safe_metrics(val_metrics)
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                seed=seed,
                config=config,
                val_metrics=val_metrics,
                dataset_sha256=dataset_hash,
            )

        append_history(
            history_path,
            [
                epoch,
                f"{train_metrics['loss']:.8f}",
                f"{train_metrics['accuracy_pct']:.6f}",
                f"{val_metrics['loss']:.8f}",
                f"{val_metrics['accuracy_pct']:.6f}",
                f"{val_metrics['macro_auc_pct']:.6f}",
                f"{optimizer.param_groups[0]['lr']:.10f}",
                f"{elapsed:.3f}",
                int(selected),
            ],
        )
        print(
            f"epoch {epoch:03d}/{config.epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy_pct']:.2f}% | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy_pct']:.2f}% "
            f"macro-AUC {val_metrics['macro_auc_pct']:.3f}% | "
            f"{'SELECTED' if selected else ''}"
        )

    if best_epoch < 0 or not best_path.is_file():
        raise RuntimeError("No finite validation macro-AUC checkpoint was selected")

    # Save the terminal training state for audit/resume, but never use it for test selection.
    terminal_val = val_metrics
    save_checkpoint(
        last_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=config.epochs,
        seed=seed,
        config=config,
        val_metrics=terminal_val,
        dataset_sha256=dataset_hash,
    )

    selected_model, selected_checkpoint = load_checkpoint_model(best_path, device)
    test_metrics = evaluate(
        selected_model,
        test_loader,
        criterion,
        device,
        amp_enabled=amp_enabled,
        control=config.control,
        description="test (selected checkpoint only)",
    )

    np.savez_compressed(
        seed_dir / "test_predictions.npz",
        logits=test_metrics["logits"].numpy(),
        probabilities=test_metrics["probabilities"],
        predictions=test_metrics["predictions"],
        labels=test_metrics["labels"].numpy(),
    )

    summary = {
        "seed": seed,
        "control": config.control,
        "parameter_count": count_trainable_parameters(selected_model),
        "model_config": selected_model.model_config(),
        "token_geometry_224": selected_model.token_geometry(224),
        "selected_epoch": best_epoch,
        "selection_metric": "validation_macro_auc",
        "best_validation": best_val_metrics,
        "test": json_safe_metrics(test_metrics),
        "split_sizes": split_sizes,
        "dataset_sha256": dataset_hash,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": sha256_file(last_path),
        "environment": environment_manifest(device),
        "git_commit": selected_checkpoint.get("git_commit"),
    }
    with (seed_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def aggregate_runs(results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    accuracy = [float(result["test"]["accuracy_pct"]) for result in results]
    macro_auc = [float(result["test"]["macro_auc_pct"]) for result in results]
    aggregate = {
        "control": results[0]["control"] if results else None,
        "seeds": [result["seed"] for result in results],
        "test_accuracy_pct": t_interval_95(accuracy),
        "test_macro_auc_pct": t_interval_95(macro_auc),
        "selected_epochs": [result["selected_epoch"] for result in results],
        "runs": results,
    }
    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, ensure_ascii=False)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IDENGATE with validation-only checkpoint selection"
    )
    parser.add_argument("--npz", required=True, help="Path to a MedMNIST-format NPZ file")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of target classes")
    parser.add_argument("--output-dir", default="outputs/retinamnist_primary")
    parser.add_argument("--control", choices=VALID_CONTROLS, default="full")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--stem-stride", type=int, default=4)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--no-horizontal-flip", action="store_true")
    parser.add_argument("--crop-padding", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_classes < 2:
        raise ValueError("--num-classes must be at least 2")
    if args.control == "shuffle_mgf" and args.data_parallel:
        raise ValueError("Shuffle-MGF requires a single process/device so the permutation spans the full mini-batch")
    validate_reported_parameter_counts()
    npz_path = str(Path(args.npz).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        npz_path=npz_path,
        output_dir=str(output_dir),
        control=args.control,
        num_classes=args.num_classes,
        image_size=args.image_size,
        stem_stride=args.stem_stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        crop_padding=args.crop_padding,
        horizontal_flip=not args.no_horizontal_flip,
        use_amp=not args.no_amp,
        deterministic=not args.non_deterministic,
        allow_tf32=not args.disable_tf32,
        data_parallel=args.data_parallel,
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=False)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    results = [run_seed(config, seed, device) for seed in args.seeds]
    aggregate = aggregate_runs(results, output_dir)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
