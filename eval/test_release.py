from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import (
    CMANet,
    PAPER_DEPTH_PARAMETER_COUNTS,
    PAPER_NO_MGF_PARAMETER_COUNT,
    build_primary_model,
    count_trainable_parameters,
)


def test_parameter_counts() -> None:
    for depth, expected in PAPER_DEPTH_PARAMETER_COUNTS.items():
        assert count_trainable_parameters(CMANet(depth=depth)) == expected
    assert count_trainable_parameters(CMANet(use_mgf=False)) == PAPER_NO_MGF_PARAMETER_COUNT


def test_224_geometry_is_feasible_and_explicit() -> None:
    model = build_primary_model(stem_stride=4)
    assert model.token_geometry(224) == [(56, 56), (28, 28), (14, 14)]


def test_exact_identity_equivalence() -> None:
    torch.manual_seed(5)
    model = build_primary_model().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        alpha_zero = model(x, alpha=0.0)
        all_identity = model(x, alpha=0.1, identity_roles=("Q", "K", "V"))
    torch.testing.assert_close(alpha_zero, all_identity, rtol=0.0, atol=0.0)


def test_gate_bounds() -> None:
    torch.manual_seed(7)
    model = build_primary_model().eval()
    # Move gates away from their all-one initialization while preserving bounds.
    for module in model.modules():
        if hasattr(module, "gq") and getattr(module, "gq") is not None:
            for name in ("gq", "gk", "gv"):
                gate = getattr(module, name)
                torch.nn.init.normal_(gate.weight, mean=0.0, std=0.2)
                torch.nn.init.normal_(gate.bias, mean=0.0, std=0.2)
    x = torch.randn(3, 3, 32, 32)
    with torch.inference_mode():
        _, stages = model(x, alpha=0.1, return_gates=True)
    for stage in stages:
        for role in ("Q", "K", "V"):
            assert torch.all(stage[role] >= 0.9)
            assert torch.all(stage[role] <= 1.1)


def test_selected_intervention_only_replaces_selected_role() -> None:
    torch.manual_seed(11)
    model = build_primary_model().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        _, full = model(x, alpha=0.1, return_gates=True)
        _, q_identity = model(x, alpha=0.1, identity_roles=("Q",), return_gates=True)
    for full_stage, changed_stage in zip(full, q_identity):
        assert torch.equal(changed_stage["Q"], torch.ones_like(changed_stage["Q"]))
        assert torch.equal(full_stage["K"], changed_stage["K"])
        assert torch.equal(full_stage["V"], changed_stage["V"])


def test_shuffle_preserves_joint_batch_multiset() -> None:
    torch.manual_seed(13)
    model = build_primary_model().eval()
    x = torch.randn(4, 3, 32, 32)
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
            torch.testing.assert_close(
                shuffled_stage[role],
                full_stage[role].index_select(0, permutation),
                rtol=0.0,
                atol=0.0,
            )


def test_no_mgf_has_no_gate_parameters() -> None:
    model = CMANet(use_mgf=False)
    names = [name for name, _ in model.named_parameters()]
    assert not any(".gq." in name or ".gk." in name or ".gv." in name for name in names)
