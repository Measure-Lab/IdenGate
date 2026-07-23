"""Verify whether a checkpoint matches the paper-constrained IDENGATE model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from model import CMANet, count_trainable_parameters


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def extract_state_dict(payload: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a state-dict-like mapping")
    if "model_state_dict" in payload:
        return strip_module_prefix(payload["model_state_dict"]), payload.get("model_config")
    if "state_dict" in payload:
        return strip_module_prefix(payload["state_dict"]), payload.get("model_config")
    if all(isinstance(value, torch.Tensor) for value in payload.values()):
        return strip_module_prefix(payload), None
    raise KeyError("Could not locate model_state_dict/state_dict")


def compare_shapes(model: CMANet, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    expected = model.state_dict()
    missing = sorted(set(expected).difference(state_dict))
    unexpected = sorted(set(state_dict).difference(expected))
    mismatched = []
    for key in sorted(set(expected).intersection(state_dict)):
        if tuple(expected[key].shape) != tuple(state_dict[key].shape):
            mismatched.append(
                {
                    "key": key,
                    "expected": tuple(expected[key].shape),
                    "observed": tuple(state_dict[key].shape),
                }
            )
    return {"missing": missing, "unexpected": unexpected, "shape_mismatches": mismatched}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify an IDENGATE checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stem-stride", type=int, default=4)
    parser.add_argument("--no-mgf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, stored_config = extract_state_dict(payload)

    if stored_config is not None:
        model = CMANet(**stored_config)
    else:
        model = CMANet(use_mgf=not args.no_mgf, stem_stride=args.stem_stride)

    report = compare_shapes(model, state_dict)
    print(f"checkpoint: {checkpoint_path}")
    print(f"model parameters: {count_trainable_parameters(model):,}")
    print(f"stored model config: {stored_config}")
    print(f"missing keys: {len(report['missing'])}")
    print(f"unexpected keys: {len(report['unexpected'])}")
    print(f"shape mismatches: {len(report['shape_mismatches'])}")

    if report["missing"]:
        print("\nMissing keys:")
        for key in report["missing"]:
            print(f"  {key}")
    if report["unexpected"]:
        print("\nUnexpected keys:")
        for key in report["unexpected"]:
            print(f"  {key}")
    if report["shape_mismatches"]:
        print("\nShape mismatches:")
        for item in report["shape_mismatches"]:
            print(f"  {item['key']}: expected {item['expected']}, observed {item['observed']}")

    if any(report.values()):
        raise SystemExit("Checkpoint is not strictly compatible with this paper-constrained model.")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        alpha_zero = model(x, alpha=0.0)
        qkv_identity = model(x, alpha=0.1, identity_roles=("Q", "K", "V"))
    torch.testing.assert_close(alpha_zero, qkv_identity, rtol=0.0, atol=0.0)
    print("Strict state-dict compatibility: PASS")
    print("Exact alpha=0 / QKV-identity equivalence: PASS")
    print(f"224x224 token geometry: {model.token_geometry(224)}")


if __name__ == "__main__":
    main()
