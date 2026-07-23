"""Official IDENGATE model definition used by the public reproduction code.

The module implements the architecture and intervention semantics described in

    IDENGATE: Bounded Role-Specific Preprojection Gating with Exact Identity
    Control for Medical Image Classification.

The three-block RetinaMNIST configuration has exactly 2,353,377 trainable
parameters.  The separately instantiated no-MGF control has 2,350,689.

Terminology
-----------
MGF ON
    Learned-gate state of a trained IDENGATE checkpoint, alpha=0.1.
MGF OFF
    Exact identity-gate state of the same checkpoint, alpha=0.
no-MGF
    Separately trained architecture in which MGF parameters do not exist.
Shuffle-MGF
    Mini-batch gate tuples are permuted jointly across Q/K/V, preserving the
    mini-batch multiset while breaking image-gate correspondence.

The class/module names intentionally retain compatibility with the early public
scripts (CMANet, CMABlock, stage1/stage2/stage3, encoder, gq/gk/gv).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn as nn

VALID_ROLES = ("Q", "K", "V")
PAPER_PRIMARY_PARAMETER_COUNT = 2_353_377
PAPER_NO_MGF_PARAMETER_COUNT = 2_350_689
PAPER_DEPTH_PARAMETER_COUNTS = {1: 115_913, 2: 565_457, 3: 2_353_377}


def _normalize_roles(roles: Iterable[str] | None) -> frozenset[str]:
    if roles is None:
        return frozenset()
    normalized = frozenset(str(role).upper() for role in roles)
    invalid = normalized.difference(VALID_ROLES)
    if invalid:
        raise ValueError(f"Unsupported identity role(s): {sorted(invalid)}")
    return normalized


@dataclass(frozen=True)
class GateState:
    """Inference-time gate state."""

    alpha: float = 0.1
    identity_roles: frozenset[str] = frozenset()
    shuffle: bool = False

    @classmethod
    def build(
        cls,
        *,
        alpha: float,
        identity_roles: Iterable[str] | None = None,
        shuffle: bool = False,
    ) -> "GateState":
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        return cls(
            alpha=float(alpha),
            identity_roles=_normalize_roles(identity_roles),
            shuffle=bool(shuffle),
        )


class ALME(nn.Module):
    """Adaptive Local Mapping Encoder (manuscript Eq. 6).

    Z     = BN(DWConv3x3(F))
    s     = sigmoid(h(GAP(Z)))
    F_hat = Conv1x1(Z * s)
    m     = GAP(F_hat)

    Biases are disabled in the depthwise and final pointwise convolutions.
    Together with an MLP ratio of two in CSA, this is required by the exact
    parameter counts reported in Fig. 3 of the manuscript.
    """

    def __init__(self, dim: int, reduction: int = 16) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")

        hidden = max(1, dim // reduction)
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(dim)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"ALME expects [B,C,H,W], received {tuple(x.shape)}")
        z = self.bn(self.dwconv(x))
        s = self.se(z)
        f_hat = self.proj(z * s)
        descriptor = self.gap(f_hat).flatten(1)
        return f_hat, descriptor


class ConditionedSelfAttention(nn.Module):
    """Conditioned Self-Attention with role-specific preprojection gates.

    The public attribute names ``encoder``, ``gq``, ``gk`` and ``gv`` are kept
    compatible with the original implementation.  The depthwise 1x1 layers on
    a pooled descriptor implement the per-channel affine maps

        g_r(m) = a_r * m + b_r.

    Multiplication of token channels occurs before MultiheadAttention applies
    its learned Q/K/V projection matrices.
    """

    def __init__(
        self,
        dim: int,
        nhead: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        use_mgf: bool = True,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if dim % nhead != 0:
            raise ValueError(f"dim={dim} must be divisible by nhead={nhead}")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

        self.dim = int(dim)
        self.use_mgf = bool(use_mgf)
        self.max_tokens = int(max_tokens)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * mlp_ratio,
            dropout=dropout,
            activation="relu",
            batch_first=False,
            norm_first=False,
        )

        if self.use_mgf:
            self.gq = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)
            self.gk = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)
            self.gv = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)
            for module in (self.gq, self.gk, self.gv):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        else:
            self.gq = None
            self.gk = None
            self.gv = None

    @staticmethod
    def _affine(descriptor: torch.Tensor, module: nn.Conv2d) -> torch.Tensor:
        return module(descriptor[:, :, None, None]).flatten(1)

    def _compute_gates(
        self,
        descriptor: torch.Tensor,
        state: GateState,
        permutation: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch_size, channels = descriptor.shape
        if channels != self.dim:
            raise ValueError(
                f"descriptor has {channels} channels, expected {self.dim}"
            )

        if not self.use_mgf:
            ones = torch.ones(
                batch_size,
                channels,
                device=descriptor.device,
                dtype=descriptor.dtype,
            )
            return {role: ones for role in VALID_ROLES}

        assert self.gq is not None and self.gk is not None and self.gv is not None
        gates = {
            "Q": 1.0 + state.alpha * torch.tanh(self._affine(descriptor, self.gq)),
            "K": 1.0 + state.alpha * torch.tanh(self._affine(descriptor, self.gk)),
            "V": 1.0 + state.alpha * torch.tanh(self._affine(descriptor, self.gv)),
        }

        if state.shuffle:
            if permutation is None:
                permutation = torch.randperm(batch_size, device=descriptor.device)
            if permutation.ndim != 1 or permutation.numel() != batch_size:
                raise ValueError("permutation must contain one index per batch item")
            for role in VALID_ROLES:
                gates[role] = gates[role].index_select(0, permutation)

        for role in state.identity_roles:
            gates[role] = torch.ones_like(gates[role])
        return gates

    def _gated_encoder_forward(
        self,
        seq: torch.Tensor,
        gates: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        # seq: [L,B,C], gates: [B,C]
        q = seq * gates["Q"].unsqueeze(0)
        k = seq * gates["K"].unsqueeze(0)
        v = seq * gates["V"].unsqueeze(0)

        enc = self.encoder
        attn_out, _ = enc.self_attn(q, k, v, need_weights=False)
        src = enc.norm1(seq + enc.dropout1(attn_out))
        ffn = enc.linear2(enc.dropout(enc.activation(enc.linear1(src))))
        return enc.norm2(src + enc.dropout2(ffn))

    def forward(
        self,
        feature_map: torch.Tensor,
        descriptor: torch.Tensor,
        *,
        state: GateState,
        permutation: torch.Tensor | None = None,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [B,C,H,W]")
        batch_size, channels, height, width = feature_map.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, received {channels}")
        token_count = height * width
        if self.max_tokens > 0 and token_count > self.max_tokens:
            raise RuntimeError(
                f"CSA received {token_count:,} tokens ({height}x{width}), exceeding "
                f"the configured safety limit {self.max_tokens:,}. Check the stem "
                "stride or tokenization geometry before training."
            )

        seq = feature_map.flatten(2).permute(2, 0, 1).contiguous()
        gates = self._compute_gates(descriptor, state, permutation)

        if self.use_mgf:
            output = self._gated_encoder_forward(seq, gates)
        else:
            # Separately trained no-MGF model: no gate parameters are present.
            output = self.encoder(seq)

        output_map = output.permute(1, 2, 0).reshape(
            batch_size, channels, height, width
        )
        if return_gates:
            return output_map, gates
        return output_map


class CMABlock(nn.Module):
    """One ALME-MGF-CSA block.

    The outer residual ``x + CSA(ALME(x))`` is retained from the original
    experimental implementation.  CSA itself also retains the standard
    attention and feed-forward residual paths.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 16,
        nhead: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        use_mgf: bool = True,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        self.alme = ALME(dim, reduction=reduction)
        self.attn = ConditionedSelfAttention(
            dim=dim,
            nhead=nhead,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_mgf=use_mgf,
            max_tokens=max_tokens,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: GateState,
        permutation: torch.Tensor | None = None,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        f_hat, descriptor = self.alme(x)
        csa = self.attn(
            f_hat,
            descriptor,
            state=state,
            permutation=permutation,
            return_gates=return_gates,
        )
        if return_gates:
            output, gates = csa
            return x + output, gates
        return x + csa


class CMANet(nn.Module):
    """IDENGATE classifier retaining checkpoint-compatible public names.

    The default stem uses stride 4.  With 224x224 inputs, CSA sees token grids
    56x56, 28x28 and 14x14, followed by a 7x7 classifier feature map.  The
    stride is parameter-free but is stored in every new checkpoint because it
    is essential for numerical reproducibility.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        *,
        depth: int = 3,
        base_dim: int = 64,
        reduction: int = 16,
        nhead: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        alpha: float = 0.1,
        use_mgf: bool = True,
        stem_stride: int = 4,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if depth not in (1, 2, 3):
            raise ValueError("depth must be one of {1,2,3}")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if stem_stride <= 0:
            raise ValueError("stem_stride must be positive")

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.depth = int(depth)
        self.base_dim = int(base_dim)
        self.reduction = int(reduction)
        self.nhead = int(nhead)
        self.mlp_ratio = int(mlp_ratio)
        self.dropout = float(dropout)
        self.alpha = float(alpha)
        self.use_mgf = bool(use_mgf)
        self.stem_stride = int(stem_stride)
        self.max_tokens = int(max_tokens)

        channels = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels[0],
                kernel_size=3,
                stride=stem_stride,
                padding=1,
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        def make_stage(stage_index: int) -> nn.Sequential:
            dim = channels[stage_index]
            return nn.Sequential(
                CMABlock(
                    dim,
                    reduction=reduction,
                    nhead=nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_mgf=use_mgf,
                    max_tokens=max_tokens,
                ),
                nn.Conv2d(
                    dim,
                    channels[stage_index + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(channels[stage_index + 1]),
                nn.ReLU(inplace=True),
            )

        self.stage1 = make_stage(0)
        self.stage2 = make_stage(1) if depth >= 2 else None
        self.stage3 = make_stage(2) if depth >= 3 else None
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[depth], num_classes)

    def model_config(self) -> dict[str, int | float | bool]:
        return {
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "depth": self.depth,
            "base_dim": self.base_dim,
            "reduction": self.reduction,
            "nhead": self.nhead,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "alpha": self.alpha,
            "use_mgf": self.use_mgf,
            "stem_stride": self.stem_stride,
            "max_tokens": self.max_tokens,
        }

    @staticmethod
    def _forward_stage_tail(stage: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        for module in list(stage.children())[1:]:
            x = module(x)
        return x

    def _forward_stage(
        self,
        stage: nn.Sequential,
        x: torch.Tensor,
        *,
        state: GateState,
        permutation: torch.Tensor | None,
        return_gates: bool,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor] | None]:
        block = stage[0]
        assert isinstance(block, CMABlock)
        if return_gates:
            x, gates = block(
                x,
                state=state,
                permutation=permutation,
                return_gates=True,
            )
        else:
            x = block(
                x,
                state=state,
                permutation=permutation,
                return_gates=False,
            )
            gates = None
        x = self._forward_stage_tail(stage, x)
        return x, gates

    def token_geometry(self, input_size: int = 224) -> list[tuple[int, int]]:
        """Return pre-attention spatial grids for a square input."""
        # Conv output: floor((N + 2P - K)/S) + 1, K=3,P=1.
        size = (input_size + 2 - 3) // self.stem_stride + 1
        geometry: list[tuple[int, int]] = []
        for _ in range(self.depth):
            geometry.append((size, size))
            size = (size + 2 - 3) // 2 + 1
        return geometry

    def forward(
        self,
        x: torch.Tensor,
        *,
        alpha: float | None = None,
        identity_roles: Iterable[str] | None = None,
        shuffle_gates: bool = False,
        gate_permutation: torch.Tensor | None = None,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[Mapping[str, torch.Tensor]]]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], received {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, received {x.shape[1]}"
            )

        state = GateState.build(
            alpha=self.alpha if alpha is None else float(alpha),
            identity_roles=identity_roles,
            shuffle=shuffle_gates,
        )
        if state.shuffle and gate_permutation is None:
            gate_permutation = torch.randperm(x.shape[0], device=x.device)

        x = self.stem(x)
        all_gates: list[Mapping[str, torch.Tensor]] = []
        stages = [self.stage1]
        if self.stage2 is not None:
            stages.append(self.stage2)
        if self.stage3 is not None:
            stages.append(self.stage3)

        for stage in stages:
            x, gates = self._forward_stage(
                stage,
                x,
                state=state,
                permutation=gate_permutation,
                return_gates=return_gates,
            )
            if gates is not None:
                all_gates.append(gates)

        logits = self.fc(self.gap(x).flatten(1))
        if return_gates:
            return logits, all_gates
        return logits

    @torch.no_grad()
    def mgf_on(self, x: torch.Tensor) -> torch.Tensor:
        return self(x, alpha=self.alpha)

    @torch.no_grad()
    def mgf_off(self, x: torch.Tensor) -> torch.Tensor:
        return self(x, alpha=0.0)

    @torch.no_grad()
    def intervene(self, x: torch.Tensor, roles: Iterable[str]) -> torch.Tensor:
        return self(x, alpha=self.alpha, identity_roles=roles)


# Publication-facing names.
IDENGATE = CMANet
IDENGATEBlock = CMABlock


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_primary_model(
    *,
    num_classes: int = 5,
    in_channels: int = 3,
    stem_stride: int = 4,
) -> CMANet:
    return CMANet(
        num_classes=num_classes,
        in_channels=in_channels,
        depth=3,
        base_dim=64,
        reduction=16,
        nhead=4,
        mlp_ratio=2,
        dropout=0.1,
        alpha=0.1,
        use_mgf=True,
        stem_stride=stem_stride,
        max_tokens=4096,
    )


def validate_reported_parameter_counts() -> None:
    observed = {
        depth: count_trainable_parameters(CMANet(depth=depth))
        for depth in (1, 2, 3)
    }
    if observed != PAPER_DEPTH_PARAMETER_COUNTS:
        raise RuntimeError(
            f"Depth parameter counts do not match the paper: {observed}"
        )
    no_mgf = count_trainable_parameters(CMANet(use_mgf=False))
    if no_mgf != PAPER_NO_MGF_PARAMETER_COUNT:
        raise RuntimeError(
            f"no-MGF parameter count {no_mgf:,} != {PAPER_NO_MGF_PARAMETER_COUNT:,}"
        )


if __name__ == "__main__":
    validate_reported_parameter_counts()
    model = build_primary_model()
    print(f"IDENGATE parameters: {count_trainable_parameters(model):,}")
    print(f"no-MGF parameters:   {count_trainable_parameters(CMANet(use_mgf=False)):,}")
    print(f"224x224 token grids: {model.token_geometry(224)}")
