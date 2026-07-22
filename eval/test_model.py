"""Integrity tests for the public IDENGATE model implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import IDENGATE, build_primary_model, count_trainable_parameters


def test_reported_parameter_counts() -> None:
    assert count_trainable_parameters(build_primary_model()) == 2_353_377
    assert count_trainable_parameters(IDENGATE(use_mgf=False)) == 2_350_689


def test_exact_same_checkpoint_identity() -> None:
    torch.manual_seed(7)
    model = build_primary_model().eval()
    x = torch.randn(2, 3, 16, 16)

    with torch.inference_mode():
        alpha_zero = model(x, alpha=0.0)
        all_roles_identity = model(x, alpha=0.1, identity_roles=("Q", "K", "V"))

    torch.testing.assert_close(alpha_zero, all_roles_identity, rtol=0.0, atol=0.0)


def test_primary_gate_bounds() -> None:
    torch.manual_seed(11)
    model = build_primary_model().eval()
    x = torch.randn(2, 3, 16, 16)

    with torch.inference_mode():
        _, gate_stages = model(x, alpha=0.1, return_gates=True)

    for stage in gate_stages:
        for role in ("Q", "K", "V"):
            assert torch.all(stage[role] >= 0.9)
            assert torch.all(stage[role] <= 1.1)


def test_selected_role_intervention_only_changes_selected_gate() -> None:
    torch.manual_seed(13)
    model = build_primary_model().eval()
    x = torch.randn(2, 3, 16, 16)

    with torch.inference_mode():
        _, full = model(x, alpha=0.1, return_gates=True)
        _, q_identity = model(x, alpha=0.1, identity_roles=("Q",), return_gates=True)

    for full_stage, intervention_stage in zip(full, q_identity):
        assert torch.equal(intervention_stage["Q"], torch.ones_like(intervention_stage["Q"]))
        assert torch.equal(full_stage["K"], intervention_stage["K"])
        assert torch.equal(full_stage["V"], intervention_stage["V"])


def test_shuffle_preserves_gate_multiset() -> None:
    torch.manual_seed(17)
    model = build_primary_model().eval()
    x = torch.randn(4, 3, 16, 16)
    permutation = torch.tensor([2, 0, 3, 1])

    with torch.inference_mode():
        _, full = model(x, alpha=0.1, return_gates=True)
        _, shuffled = model(
            x,
            alpha=0.1,
            shuffle_gates=True,
            gate_permutation=permutation,
            return_gates=True,
        )

    for full_stage, shuffled_stage in zip(full, shuffled):
        for role in ("Q", "K", "V"):
            expected = full_stage[role].index_select(0, permutation)
            torch.testing.assert_close(expected, shuffled_stage[role], rtol=0.0, atol=0.0)
