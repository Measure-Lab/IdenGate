"""Fixed-checkpoint role-specific identity interventions for IDENGATE.

This script reproduces the protocol underlying Table II of the manuscript.
For each seed-matched Full checkpoint, selected Q/K/V gates are replaced by
exact identity at inference. Inputs, model parameters, normalization
statistics, ALME/descriptor computations, and all non-selected gates remain
fixed.

For each state:

* accuracy and one-vs-rest macro-AUC use uncalibrated test probabilities;
* a scalar temperature is fitted to that state's validation logits by
  negative-log-likelihood minimization;
* 10-bin ECE is evaluated on temperature-scaled test probabilities;
* AURC uses raw maximum-softmax confidence;
* prediction changes are measured relative to the Full state;
* deltas and nominal paired 95% CIs are computed across seeds 42-46.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import IDENGATE, count_trainable_parameters
from train import (
    EXPECTED_PRIMARY_PARAMS,
    PAPER_SEEDS,
    TrainConfig,
    build_dataloaders,
    compute_macro_auc,
    set_global_seed,
)


STATES: dict[str, tuple[str, ...]] = {
    "full": (),
    "q_identity": ("Q",),
    "k_identity": ("K",),
    "v_identity": ("V",),
    "qk_identity": ("Q", "K"),
    "qkv_identity": ("Q", "K", "V"),
}

DISPLAY_NAMES = {
    "full": "None (Full)",
    "q_identity": "Q",
    "k_identity": "K",
    "v_identity": "V",
    "qk_identity": "Q, K",
    "qkv_identity": "Q, K, V",
}


@torch.inference_mode()
def collect_logits(
    model: IDENGATE,
    loader: DataLoader,
    device: torch.device,
    *,
    alpha: float,
    identity_roles: Iterable[str],
    use_amp: bool,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for images, labels in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, alpha=alpha, identity_roles=identity_roles)
        logits_parts.append(logits.float().cpu())
        label_parts.append(labels.cpu())
    return torch.cat(logits_parts, dim=0), torch.cat(label_parts, dim=0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one positive temperature on validation logits."""

    logits = logits.detach().double()
    labels = labels.detach().long()
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
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
    n_bins: int = 10,
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not np.any(mask):
            continue
        weight = float(mask.mean())
        ece += weight * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return ece


def area_under_risk_coverage(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    errors = (prediction != labels).astype(np.float64)
    order = np.argsort(-confidence, kind="mergesort")
    cumulative_errors = np.cumsum(errors[order])
    retained = np.arange(1, labels.size + 1, dtype=np.float64)
    risk = cumulative_errors / retained
    coverage = retained / float(labels.size)
    return float(np.trapz(risk, coverage))


def state_metrics(
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> dict[str, Any]:
    raw_probabilities = torch.softmax(raw_logits, dim=1).numpy()
    calibrated_probabilities = torch.softmax(raw_logits / temperature, dim=1).numpy()
    labels_np = labels.numpy()
    predictions = raw_probabilities.argmax(axis=1)
    accuracy = 100.0 * float(np.mean(predictions == labels_np))
    macro_auc = compute_macro_auc(labels_np, raw_probabilities)
    return {
        "accuracy_pct": accuracy,
        "macro_auc_pct": 100.0 * macro_auc,
        "ece": expected_calibration_error(calibrated_probabilities, labels_np, n_bins=10),
        "aurc": area_under_risk_coverage(raw_probabilities, labels_np),
        "temperature": temperature,
        "predictions": predictions,
    }


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[IDENGATE, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["model_config"])
    if not model_config.get("use_mgf", False):
        raise ValueError(f"Identity intervention requires a Full MGF checkpoint: {checkpoint_path}")
    model = IDENGATE(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if count_trainable_parameters(model) != EXPECTED_PRIMARY_PARAMS:
        raise RuntimeError("Loaded model does not match the reported primary parameter count")
    model.to(device).eval()
    return model, checkpoint


def run_seed(
    npz_path: str,
    checkpoint_path: Path,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> dict[str, Any]:
    set_global_seed(seed, deterministic=True, allow_tf32=True)
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    alpha = float(checkpoint["model_config"].get("alpha", 0.1))

    data_config = TrainConfig(
        npz_path=npz_path,
        output_dir=".",
        batch_size=batch_size,
        num_workers=num_workers,
        alpha=alpha,
    )
    _, val_loader, test_loader, split_sizes = build_dataloaders(data_config, seed)

    results: dict[str, Any] = {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "selected_epoch": checkpoint.get("selected_epoch"),
        "split_sizes": split_sizes,
        "states": {},
    }

    full_predictions: np.ndarray | None = None
    qkv_test_logits: torch.Tensor | None = None
    qkv_test_labels: torch.Tensor | None = None

    for state_name, identity_roles in STATES.items():
        val_logits, val_labels = collect_logits(
            model,
            val_loader,
            device,
            alpha=alpha,
            identity_roles=identity_roles,
            use_amp=use_amp,
            description=f"seed {seed} {state_name} validation",
        )
        test_logits, test_labels = collect_logits(
            model,
            test_loader,
            device,
            alpha=alpha,
            identity_roles=identity_roles,
            use_amp=use_amp,
            description=f"seed {seed} {state_name} test",
        )
        temperature = fit_temperature(val_logits, val_labels)
        metrics = state_metrics(test_logits, test_labels, temperature)

        if full_predictions is None:
            full_predictions = metrics["predictions"]
            prediction_changed = 0.0
        else:
            prediction_changed = 100.0 * float(np.mean(metrics["predictions"] != full_predictions))

        results["states"][state_name] = {
            key: value for key, value in metrics.items() if key != "predictions"
        }
        results["states"][state_name]["prediction_changed_pct"] = prediction_changed

        if state_name == "qkv_identity":
            qkv_test_logits = test_logits
            qkv_test_labels = test_labels

    # Verify the paper's exact identity equivalence: all-role identity at
    # alpha=0.1 must match alpha=0 in the same checkpoint.
    alpha0_logits, alpha0_labels = collect_logits(
        model,
        test_loader,
        device,
        alpha=0.0,
        identity_roles=(),
        use_amp=use_amp,
        description=f"seed {seed} exact alpha=0 verification",
    )
    if qkv_test_logits is None or qkv_test_labels is None:
        raise RuntimeError("qkv identity state was not evaluated")
    if not torch.equal(qkv_test_labels, alpha0_labels):
        raise RuntimeError("Label order changed during paired identity verification")
    torch.testing.assert_close(qkv_test_logits, alpha0_logits, rtol=0.0, atol=0.0)
    results["exact_identity_verified"] = True
    return results


def paired_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size <= 1:
        return mean, mean, mean
    std = float(values.std(ddof=1))
    t_critical = 2.7764451051977987 if values.size == 5 else 1.959963984540054
    half_width = t_critical * std / math.sqrt(values.size)
    return mean, mean - half_width, mean + half_width


def summarize(seed_results: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full_accuracy = np.array([
        result["states"]["full"]["accuracy_pct"] for result in seed_results
    ])
    full_auc = np.array([
        result["states"]["full"]["macro_auc_pct"] for result in seed_results
    ])

    summary_json: dict[str, Any] = {"states": {}, "seeds": [r["seed"] for r in seed_results]}

    for state_name in STATES:
        state_values = [result["states"][state_name] for result in seed_results]
        accuracy = np.array([value["accuracy_pct"] for value in state_values])
        macro_auc = np.array([value["macro_auc_pct"] for value in state_values])
        ece = np.array([value["ece"] for value in state_values])
        aurc = np.array([value["aurc"] for value in state_values])
        prediction_changed = np.array([value["prediction_changed_pct"] for value in state_values])

        delta_accuracy = accuracy - full_accuracy
        delta_auc = macro_auc - full_auc
        dacc_mean, dacc_low, dacc_high = paired_ci(delta_accuracy)
        dauc_mean, dauc_low, dauc_high = paired_ci(delta_auc)

        row = {
            "state": DISPLAY_NAMES[state_name],
            "accuracy_mean_pct": accuracy.mean(),
            "accuracy_sd_pct": accuracy.std(ddof=1),
            "delta_accuracy_mean_pp": dacc_mean,
            "delta_accuracy_ci95_low_pp": dacc_low,
            "delta_accuracy_ci95_high_pp": dacc_high,
            "macro_auc_mean_pct": macro_auc.mean(),
            "macro_auc_sd_pct": macro_auc.std(ddof=1),
            "delta_macro_auc_mean_pp": dauc_mean,
            "delta_macro_auc_ci95_low_pp": dauc_low,
            "delta_macro_auc_ci95_high_pp": dauc_high,
            "ece_mean": ece.mean(),
            "ece_sd": ece.std(ddof=1),
            "aurc_mean": aurc.mean(),
            "aurc_sd": aurc.std(ddof=1),
            "prediction_changed_mean_pct": prediction_changed.mean(),
            "prediction_changed_sd_pct": prediction_changed.std(ddof=1),
        }
        rows.append(row)
        summary_json["states"][state_name] = row

    return pd.DataFrame(rows), summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce IDENGATE Table II interventions")
    parser.add_argument("--npz", required=True, help="Path to retinamnist_224.npz")
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        help="Directory containing seed_42/.../best_validation_macro_auc.pt",
    )
    parser.add_argument("--output-dir", default="outputs/identity_interventions")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    use_amp = bool(not args.no_amp and device.type == "cuda")
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in args.seeds:
        checkpoint = checkpoint_root / f"seed_{seed}" / "best_validation_macro_auc.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        result = run_seed(
            npz_path=str(Path(args.npz).expanduser().resolve()),
            checkpoint_path=checkpoint,
            seed=seed,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=use_amp,
        )
        seed_results.append(result)
        with (output_dir / f"seed_{seed}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

    table, summary = summarize(seed_results)
    table.to_csv(output_dir / "table_ii_reproduced.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(table.to_string(index=False))
    print(f"\nSaved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
