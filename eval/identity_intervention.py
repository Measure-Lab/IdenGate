"""Fixed-checkpoint Q/K/V identity interventions for IDENGATE Table II.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import trapezoid
from torch.amp import autocast
from tqdm import tqdm

from model import CMANet
from train import (
    MedMNISTNPZDataset,
    classwise_ovr_auc,
    load_checkpoint_model,
    load_official_npz_splits,
    macro_auc,
    seed_worker,
    sha256_file,
)
from torch.utils.data import DataLoader

STATES: dict[str, tuple[str, ...]] = {
    "full": (),
    "q_identity": ("Q",),
    "k_identity": ("K",),
    "v_identity": ("V",),
    "qk_identity": ("Q", "K"),
    "qkv_identity": ("Q", "K", "V"),
}
DISPLAY = {
    "full": "None (Full)",
    "q_identity": "Q",
    "k_identity": "K",
    "v_identity": "V",
    "qk_identity": "Q, K",
    "qkv_identity": "Q, K, V",
}


def build_eval_loaders(
    npz_path: str,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[DataLoader, DataLoader]:
    splits = load_official_npz_splits(npz_path)
    common_dataset = {
        "image_size": image_size,
        "augment": False,
        "crop_padding": 0,
        "horizontal_flip": False,
        "mean": mean,
        "std": std,
    }
    val_set = MedMNISTNPZDataset(*splits["val"], **common_dataset)
    test_set = MedMNISTNPZDataset(*splits["test"], **common_dataset)
    common_loader: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        common_loader["prefetch_factor"] = 2
    return DataLoader(val_set, **common_loader), DataLoader(test_set, **common_loader)


@torch.inference_mode()
def collect_logits(
    model: CMANet,
    loader: DataLoader,
    device: torch.device,
    *,
    alpha: float,
    identity_roles: Iterable[str],
    amp_enabled: bool,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    for images, labels in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images, alpha=alpha, identity_roles=identity_roles)
        logits_parts.append(logits.float().cpu())
        labels_parts.append(labels.cpu())
    return torch.cat(logits_parts), torch.cat(labels_parts)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one positive scalar temperature by validation NLL minimization."""
    logits = logits.detach().clone().double()
    labels = labels.detach().clone().long()
    log_temperature = torch.zeros((), dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(1e-3, 1e3)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-3, 1e3).item())


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == 0:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def risk_coverage_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    errors = (prediction != labels).astype(np.float64)
    order = np.argsort(-confidence, kind="mergesort")
    cumulative_errors = np.cumsum(errors[order])
    coverage = np.arange(1, len(labels) + 1, dtype=np.float64) / len(labels)
    risk = cumulative_errors / np.arange(1, len(labels) + 1, dtype=np.float64)
    # Include the zero-coverage origin to make the numerical convention explicit.
    return float(trapezoid(np.concatenate([[0.0], risk]), np.concatenate([[0.0], coverage])))


def metrics_for_state(
    test_logits: torch.Tensor,
    test_labels: torch.Tensor,
    temperature: float,
) -> dict[str, Any]:
    labels = test_labels.numpy()
    raw_probabilities = torch.softmax(test_logits, dim=1).numpy()
    calibrated_probabilities = torch.softmax(test_logits / temperature, dim=1).numpy()
    predictions = raw_probabilities.argmax(axis=1)
    auc = macro_auc(labels, raw_probabilities)
    return {
        "accuracy_pct": 100.0 * float(np.mean(predictions == labels)),
        "macro_auc_pct": 100.0 * auc if np.isfinite(auc) else float("nan"),
        "classwise_auc": classwise_ovr_auc(labels, raw_probabilities),
        "temperature": temperature,
        "ece": expected_calibration_error(calibrated_probabilities, labels, bins=10),
        "aurc": risk_coverage_auc(raw_probabilities, labels),
        "predictions": predictions,
    }


def paired_t_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) <= 1:
        return mean, mean, mean
    sd = float(values.std(ddof=1))
    critical = 2.7764451051977987 if len(values) == 5 else 1.959963984540054
    half = critical * sd / math.sqrt(len(values))
    return mean, mean - half, mean + half


def run_seed(
    checkpoint_path: Path,
    npz_path: str,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    expected_hash = checkpoint.get("dataset_sha256")
    observed_hash = sha256_file(npz_path)
    if expected_hash is not None and observed_hash != expected_hash:
        raise RuntimeError("Dataset SHA-256 does not match the training checkpoint")
    if not model.use_mgf or checkpoint.get("control") != "full":
        raise ValueError("Identity intervention requires a checkpoint trained with control=full")

    train_config = checkpoint.get("train_config", {})
    image_size = int(train_config.get("image_size", 224))
    mean = tuple(train_config.get("normalization_mean", (0.5, 0.5, 0.5)))
    std = tuple(train_config.get("normalization_std", (0.5, 0.5, 0.5)))
    val_loader, test_loader = build_eval_loaders(
        npz_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        mean=mean,
        std=std,
    )

    seed = int(checkpoint["seed"])
    alpha = float(model.alpha)
    result: dict[str, Any] = {
        "seed": seed,
        "checkpoint": str(checkpoint_path.resolve()),
        "selected_epoch": int(checkpoint["epoch"]),
        "states": {},
    }
    full_predictions: np.ndarray | None = None
    qkv_logits: torch.Tensor | None = None
    qkv_labels: torch.Tensor | None = None

    for state_name, roles in STATES.items():
        val_logits, val_labels = collect_logits(
            model,
            val_loader,
            device,
            alpha=alpha,
            identity_roles=roles,
            amp_enabled=amp_enabled,
            description=f"seed {seed} {state_name} val",
        )
        test_logits, test_labels = collect_logits(
            model,
            test_loader,
            device,
            alpha=alpha,
            identity_roles=roles,
            amp_enabled=amp_enabled,
            description=f"seed {seed} {state_name} test",
        )
        temperature = fit_temperature(val_logits, val_labels)
        metrics = metrics_for_state(test_logits, test_labels, temperature)
        predictions = metrics.pop("predictions")
        if full_predictions is None:
            full_predictions = predictions
            changed = 0.0
        else:
            changed = 100.0 * float(np.mean(predictions != full_predictions))
        metrics["prediction_changed_pct"] = changed
        result["states"][state_name] = metrics
        if state_name == "qkv_identity":
            qkv_logits, qkv_labels = test_logits, test_labels

    alpha0_logits, alpha0_labels = collect_logits(
        model,
        test_loader,
        device,
        alpha=0.0,
        identity_roles=(),
        amp_enabled=amp_enabled,
        description=f"seed {seed} alpha=0 verification",
    )
    if qkv_logits is None or qkv_labels is None:
        raise RuntimeError("QKV identity state was not evaluated")
    torch.testing.assert_close(qkv_logits, alpha0_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(qkv_labels, alpha0_labels, rtol=0.0, atol=0.0)
    result["exact_identity_verified"] = True
    return result


def summarize(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_accuracy = np.array([r["states"]["full"]["accuracy_pct"] for r in seed_results])
    full_auc = np.array([r["states"]["full"]["macro_auc_pct"] for r in seed_results])
    rows: list[dict[str, Any]] = []
    for state_name in STATES:
        values = [r["states"][state_name] for r in seed_results]
        accuracy = np.array([v["accuracy_pct"] for v in values])
        auc = np.array([v["macro_auc_pct"] for v in values])
        ece = np.array([v["ece"] for v in values])
        aurc = np.array([v["aurc"] for v in values])
        changed = np.array([v["prediction_changed_pct"] for v in values])
        dacc = paired_t_ci(accuracy - full_accuracy)
        dauc = paired_t_ci(auc - full_auc)
        rows.append(
            {
                "state": DISPLAY[state_name],
                "accuracy_mean_pct": float(accuracy.mean()),
                "accuracy_sd_pct": float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
                "delta_accuracy_mean_pp": dacc[0],
                "delta_accuracy_ci95_low_pp": dacc[1],
                "delta_accuracy_ci95_high_pp": dacc[2],
                "macro_auc_mean_pct": float(auc.mean()),
                "macro_auc_sd_pct": float(auc.std(ddof=1)) if len(auc) > 1 else 0.0,
                "delta_macro_auc_mean_pp": dauc[0],
                "delta_macro_auc_ci95_low_pp": dauc[1],
                "delta_macro_auc_ci95_high_pp": dauc[2],
                "ece_mean": float(ece.mean()),
                "ece_sd": float(ece.std(ddof=1)) if len(ece) > 1 else 0.0,
                "aurc_mean": float(aurc.mean()),
                "aurc_sd": float(aurc.std(ddof=1)) if len(aurc) > 1 else 0.0,
                "prediction_changed_mean_pct": float(changed.mean()),
                "prediction_changed_sd_pct": float(changed.std(ddof=1)) if len(changed) > 1 else 0.0,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IDENGATE fixed-checkpoint identity interventions")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", default="outputs/identity_interventions")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    device = torch.device(args.device)
    amp_enabled = bool(not args.no_amp and device.type == "cuda")

    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        checkpoint_path = checkpoint_root / f"seed_{seed}" / "best_validation_macro_auc.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        result = run_seed(
            checkpoint_path,
            str(Path(args.npz).expanduser().resolve()),
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp_enabled=amp_enabled,
        )
        seed_results.append(result)
        with (output_dir / f"seed_{seed}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

    rows = summarize(seed_results)
    import csv
    with (output_dir / "table_ii_reproduced.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"seeds": args.seeds, "rows": rows}, handle, indent=2, ensure_ascii=False)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
