"""Train IDENGATE on the official RetinaMNIST NPZ splits.

The default configuration reproduces the five-seed mechanism-study protocol
reported in the manuscript:

* official train / validation / test splits;
* seeds 42, 43, 44, 45, and 46;
* 60 epochs;
* batch size 128;
* AdamW, learning rate 3e-4, weight decay 1e-4;
* cosine annealing with warm restarts (T0=10, Tmult=2);
* cross-entropy loss;
* no class weighting and no pretraining;
* checkpoint selection by the highest validation macro-AUC;
* test-set evaluation only after checkpoint selection.

The script supports the Full, no-MGF, and Shuffle-MGF configurations. MGF OFF
is not a training configuration: it is the alpha=0 identity state of a trained
Full checkpoint and is evaluated by identity_intervention.py.
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
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import IDENGATE, count_trainable_parameters


PAPER_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_PRIMARY_PARAMS = 2_353_377
EXPECTED_NO_MGF_PARAMS = 2_350_689


@dataclass(frozen=True)
class TrainConfig:
    """Complete configuration stored with every run."""

    npz_path: str
    output_dir: str
    control: str = "full"
    num_classes: int = 5
    in_channels: int = 3
    image_size: int = 224
    epochs: int = 60
    batch_size: int = 128
    num_workers: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler_t0: int = 10
    scheduler_tmult: int = 2
    alpha: float = 0.1
    depth: int = 3
    nhead: int = 4
    mlp_ratio: int = 2
    dropout: float = 0.1
    reduction: int = 16
    use_amp: bool = True
    deterministic: bool = True
    allow_tf32: bool = True
    data_parallel: bool = True
    crop_padding: int = 16
    horizontal_flip: bool = True
    normalization_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalization_std: tuple[float, float, float] = (0.5, 0.5, 0.5)


class MedMNISTNPZDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset wrapper for the official MedMNIST NPZ arrays."""

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
        self.labels = _normalize_labels(labels)

        operations: list[Any] = [transforms.ToPILImage(), transforms.Resize((image_size, image_size))]
        if augment:
            operations.append(transforms.RandomCrop(image_size, padding=crop_padding))
            if horizontal_flip:
                operations.append(transforms.RandomHorizontalFlip())
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(_ensure_three_channels),
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
            if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
                image = np.clip(image * 255.0, 0, 255)
            image = image.astype(np.uint8)

        tensor = self.transform(image)
        target = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return tensor, target


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2:
        if labels.shape[1] == 1:
            labels = labels[:, 0]
        else:
            labels = labels.argmax(axis=1)
    elif labels.ndim != 1:
        labels = labels.reshape(-1)
    return labels.astype(np.int64, copy=False)


def _ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        return tensor.repeat(3, 1, 1)
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected one or three image channels, got {tensor.shape[0]}")
    return tensor


def load_official_npz_splits(npz_path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load official train, validation, and test arrays without merging them."""

    path = Path(npz_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {
            "train_images",
            "train_labels",
            "val_images",
            "val_labels",
            "test_images",
            "test_labels",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"NPZ file is missing required arrays: {missing}")
        splits = {
            "train": (data["train_images"].copy(), data["train_labels"].copy()),
            "val": (data["val_images"].copy(), data["val_labels"].copy()),
            "test": (data["test_images"].copy(), data["test_labels"].copy()),
        }

    return splits


def set_global_seed(seed: int, deterministic: bool, allow_tf32: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    config: TrainConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    splits = load_official_npz_splits(config.npz_path)

    datasets = {
        "train": MedMNISTNPZDataset(
            *splits["train"],
            image_size=config.image_size,
            augment=True,
            crop_padding=config.crop_padding,
            horizontal_flip=config.horizontal_flip,
            mean=config.normalization_mean,
            std=config.normalization_std,
        ),
        "val": MedMNISTNPZDataset(
            *splits["val"],
            image_size=config.image_size,
            augment=False,
            crop_padding=config.crop_padding,
            horizontal_flip=False,
            mean=config.normalization_mean,
            std=config.normalization_std,
        ),
        "test": MedMNISTNPZDataset(
            *splits["test"],
            image_size=config.image_size,
            augment=False,
            crop_padding=config.crop_padding,
            horizontal_flip=False,
            mean=config.normalization_mean,
            std=config.normalization_std,
        ),
    }

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    if config.num_workers > 0:
        common["prefetch_factor"] = 2

    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(datasets["val"], shuffle=False, **common)
    test_loader = DataLoader(datasets["test"], shuffle=False, **common)
    sizes = {name: len(dataset) for name, dataset in datasets.items()}
    return train_loader, val_loader, test_loader, sizes


def build_model(config: TrainConfig) -> IDENGATE:
    if config.control not in {"full", "no_mgf", "shuffle_mgf"}:
        raise ValueError(f"Unsupported control: {config.control}")

    model = IDENGATE(
        num_classes=config.num_classes,
        in_channels=config.in_channels,
        depth=config.depth,
        base_dim=64,
        reduction=config.reduction,
        nhead=config.nhead,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        alpha=config.alpha,
        use_mgf=config.control != "no_mgf",
    )

    parameter_count = count_trainable_parameters(model)
    if config.depth == 3 and config.num_classes == 5 and config.in_channels == 3:
        expected = EXPECTED_NO_MGF_PARAMS if config.control == "no_mgf" else EXPECTED_PRIMARY_PARAMS
        if parameter_count != expected:
            raise RuntimeError(
                f"Parameter-count mismatch: expected {expected:,}, observed {parameter_count:,}"
            )
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def model_forward(model: nn.Module, images: torch.Tensor, control: str) -> torch.Tensor:
    return model(images, shuffle_gates=(control == "shuffle_mgf"))


def compute_macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """One-vs-rest macro-AUC, as defined in the manuscript."""

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


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    control: str,
    description: str,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=description, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model_forward(model, images, control)
                loss = criterion(logits, labels)

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            logits_parts.append(logits.float().cpu())
            label_parts.append(labels.cpu())

    logits = torch.cat(logits_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    probabilities = torch.softmax(logits, dim=1).numpy()
    labels_np = labels.numpy()
    predictions = probabilities.argmax(axis=1)

    accuracy = 100.0 * float(np.mean(predictions == labels_np))
    macro_auc = compute_macro_auc(labels_np, probabilities)
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy_pct": accuracy,
        "macro_auc": macro_auc,
        "macro_auc_pct": 100.0 * macro_auc if np.isfinite(macro_auc) else float("nan"),
        "logits": logits,
        "labels": labels,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    control: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model_forward(model, images, control)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy_pct": 100.0 * total_correct / max(total_examples, 1),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    }


def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"logits", "labels"}
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    seed: int,
    epoch: int,
    validation_metrics: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "model_config": {
            "num_classes": config.num_classes,
            "in_channels": config.in_channels,
            "depth": config.depth,
            "base_dim": 64,
            "reduction": config.reduction,
            "nhead": config.nhead,
            "mlp_ratio": config.mlp_ratio,
            "dropout": config.dropout,
            "alpha": config.alpha,
            "use_mgf": config.control != "no_mgf",
        },
        "training_config": asdict(config),
        "control": config.control,
        "seed": seed,
        "selected_epoch": epoch,
        "selection_metric": "validation_macro_auc",
        "validation_metrics": _json_safe_metrics(validation_metrics),
    }
    torch.save(payload, path)


def run_seed(config: TrainConfig, seed: int, device: torch.device) -> dict[str, Any]:
    set_global_seed(seed, config.deterministic, config.allow_tf32)
    seed_dir = Path(config.output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, split_sizes = build_dataloaders(config, seed)
    model = build_model(config).to(device)
    parameter_count = count_trainable_parameters(model)

    if config.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

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
    criterion = nn.CrossEntropyLoss()  # No class weighting, as reported.
    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    checkpoint_path = seed_dir / "best_validation_macro_auc.pt"
    history_path = seed_dir / "history.csv"
    best_auc = -math.inf
    best_epoch = -1
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            use_amp,
            config.control,
        )
        scheduler.step()
        val_metrics = collect_predictions(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            config.control,
            description="validation",
        )
        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy_pct": train_metrics["accuracy_pct"],
            "val_loss": val_metrics["loss"],
            "val_accuracy_pct": val_metrics["accuracy_pct"],
            "val_macro_auc": val_metrics["macro_auc"],
            "val_macro_auc_pct": val_metrics["macro_auc_pct"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": elapsed,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        current_auc = val_metrics["macro_auc"]
        if np.isfinite(current_auc) and current_auc > best_auc:
            best_auc = float(current_auc)
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model,
                config,
                seed,
                epoch,
                val_metrics,
            )

        print(
            f"seed={seed} epoch={epoch:03d}/{config.epochs} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_acc={train_metrics['accuracy_pct']:.2f}% "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_acc={val_metrics['accuracy_pct']:.2f}% "
            f"val_macro_auc={val_metrics['macro_auc_pct']:.3f}% "
            f"lr={optimizer.param_groups[0]['lr']:.8f}"
        )

    if not checkpoint_path.is_file():
        raise RuntimeError("No finite validation macro-AUC checkpoint was produced")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)

    test_metrics = collect_predictions(
        model,
        test_loader,
        criterion,
        device,
        use_amp,
        config.control,
        description="test",
    )
    result = {
        "seed": seed,
        "control": config.control,
        "selected_epoch": best_epoch,
        "parameter_count": parameter_count,
        "split_sizes": split_sizes,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test": _json_safe_metrics(test_metrics),
    }

    with (seed_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def _mean_std_ci(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    if array.size == 5:
        t_critical = 2.7764451051977987  # Student t, df=4, two-sided 95% CI.
    elif array.size > 1:
        # Normal approximation is used only for non-paper custom seed counts.
        t_critical = 1.959963984540054
    else:
        t_critical = 0.0
    half_width = t_critical * std / math.sqrt(array.size) if array.size else float("nan")
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def summarize_runs(config: TrainConfig, results: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy = [result["test"]["accuracy_pct"] for result in results]
    macro_auc = [result["test"]["macro_auc_pct"] for result in results]
    summary = {
        "control": config.control,
        "seeds": [result["seed"] for result in results],
        "n_seeds": len(results),
        "accuracy_pct": _mean_std_ci(accuracy),
        "macro_auc_pct": _mean_std_ci(macro_auc),
        "runs": results,
        "configuration": asdict(config),
        "environment": environment_manifest(),
        "npz_sha256": sha256_file(config.npz_path),
    }

    output_dir = Path(config.output_dir)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    rows = []
    for result in results:
        rows.append(
            {
                "seed": result["seed"],
                "selected_epoch": result["selected_epoch"],
                "parameter_count": result["parameter_count"],
                "test_accuracy_pct": result["test"]["accuracy_pct"],
                "test_macro_auc_pct": result["test"]["macro_auc_pct"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "seed_results.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IDENGATE using the paper-aligned RetinaMNIST protocol."
    )
    parser.add_argument("--npz", required=True, help="Path to retinamnist_224.npz")
    parser.add_argument("--output-dir", default="outputs/retinamnist_primary")
    parser.add_argument(
        "--control",
        choices=("full", "no_mgf", "shuffle_mgf"),
        default="full",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        npz_path=str(Path(args.npz).expanduser().resolve()),
        output_dir=str(output_dir.resolve()),
        control=args.control,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        data_parallel=not args.no_data_parallel,
        deterministic=not args.non_deterministic,
        allow_tf32=not args.disable_tf32,
    )

    with (output_dir / "configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=False)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    results = [run_seed(config, seed, device) for seed in args.seeds]
    summary = summarize_runs(config, results)

    print("\nCompleted paper-aligned training protocol")
    print(f"control: {config.control}")
    print(f"seeds: {summary['seeds']}")
    print(
        "test accuracy: "
        f"{summary['accuracy_pct']['mean']:.3f} ± {summary['accuracy_pct']['std']:.3f}%"
    )
    print(
        "test macro-AUC: "
        f"{summary['macro_auc_pct']['mean']:.3f} ± {summary['macro_auc_pct']['std']:.3f}%"
    )


if __name__ == "__main__":
    main()
