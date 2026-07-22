"""IDENGATE model definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn


VALID_ROLES = ("Q", "K", "V")


def _normalize_roles(roles: Iterable[str] | None) -> frozenset[str]:
    """Normalize role names and reject unsupported values."""

    if roles is None:
        return frozenset()
    normalized = frozenset(str(role).upper() for role in roles)
    invalid = normalized.difference(VALID_ROLES)
    if invalid:
        raise ValueError(f"Unsupported identity role(s): {sorted(invalid)}")
    return normalized


@dataclass(frozen=True)
class GateState:
    """Inference-time gate configuration.
    """

    alpha: float = 0.1
    identity_roles: frozenset[str] = frozenset()
    shuffle: bool = False

    @classmethod
    def from_values(
        cls,
        alpha: float = 0.1,
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
    """Adaptive Local Mapping Encoder (paper Eq. 6).
    """

    def __init__(self, dim: int, reduction: int = 16) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")

        hidden_dim = max(1, dim // reduction)
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
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
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


class ModulatedGatingFunction(nn.Module):
    """Role-specific unit-centered MGF (paper Eqs. 7-9).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.gq = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)
        self.gk = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)
        self.gv = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)

        for module in (self.gq, self.gk, self.gv):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def _affine(self, descriptor: torch.Tensor, module: nn.Conv2d) -> torch.Tensor:
        return module(descriptor[:, :, None, None]).flatten(1)

    def forward(
        self,
        descriptor: torch.Tensor,
        state: GateState,
        permutation: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if descriptor.ndim != 2 or descriptor.shape[1] != self.dim:
            raise ValueError(
                f"descriptor must have shape [B,{self.dim}], got {tuple(descriptor.shape)}"
            )

        gates = {
            "Q": 1.0 + state.alpha * torch.tanh(self._affine(descriptor, self.gq)),
            "K": 1.0 + state.alpha * torch.tanh(self._affine(descriptor, self.gk)),
            "V": 1.0 + state.alpha * torch.tanh(self._affine(descriptor, self.gv)),
        }

        if state.shuffle:
            batch_size = descriptor.shape[0]
            if permutation is None:
                permutation = torch.randperm(batch_size, device=descriptor.device)
            if permutation.ndim != 1 or permutation.numel() != batch_size:
                raise ValueError("permutation must contain one index per batch element")
            for role in VALID_ROLES:
                gates[role] = gates[role].index_select(0, permutation)

        for role in state.identity_roles:
            gates[role] = torch.ones_like(gates[role])

        return gates


class ConditionedSelfAttention(nn.Module):
    """Conditioned Self-Attention (CSA).
    """

    def __init__(
        self,
        dim: int,
        nhead: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        use_mgf: bool = True,
    ) -> None:
        super().__init__()
        if dim % nhead != 0:
            raise ValueError(f"dim={dim} must be divisible by nhead={nhead}")

        self.dim = int(dim)
        self.use_mgf = bool(use_mgf)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * mlp_ratio,
            dropout=dropout,
            activation="relu",
            batch_first=False,
            norm_first=False,
        )
        self.mgf = ModulatedGatingFunction(dim) if self.use_mgf else None

    @property
    def gq(self) -> nn.Conv2d | None:
        return None if self.mgf is None else self.mgf.gq

    @property
    def gk(self) -> nn.Conv2d | None:
        return None if self.mgf is None else self.mgf.gk

    @property
    def gv(self) -> nn.Conv2d | None:
        return None if self.mgf is None else self.mgf.gv

    def _transformer_block(
        self,
        seq: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        enc = self.encoder
        attn_out, _ = enc.self_attn(q, k, v, need_weights=False)
        src = enc.norm1(seq + enc.dropout1(attn_out))
        ffn_out = enc.linear2(enc.dropout(enc.activation(enc.linear1(src))))
        return enc.norm2(src + enc.dropout2(ffn_out))

    def forward(
        self,
        feature_map: torch.Tensor,
        descriptor: torch.Tensor,
        gate_state: GateState,
        permutation: torch.Tensor | None = None,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [B,C,H,W]")

        batch_size, channels, height, width = feature_map.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, received {channels}")

        seq = feature_map.flatten(2).permute(2, 0, 1).contiguous()  # [L,B,C]

        if self.mgf is None:
            # Separately trained no-MGF control: no gate parameters exist.
            output = self.encoder(seq)
            gates = {role: torch.ones(batch_size, channels, device=seq.device, dtype=seq.dtype)
                     for role in VALID_ROLES}
        else:
            gates = self.mgf(descriptor, gate_state, permutation=permutation)
            phi_q = gates["Q"].unsqueeze(0)
            phi_k = gates["K"].unsqueeze(0)
            phi_v = gates["V"].unsqueeze(0)
            output = self._transformer_block(
                seq=seq,
                q=seq * phi_q,
                k=seq * phi_k,
                v=seq * phi_v,
            )

        output_map = output.permute(1, 2, 0).reshape(batch_size, channels, height, width)
        if return_gates:
            return output_map, gates
        return output_map


class IDENGATEBlock(nn.Module):
    """One ALME-MGF-CSA stage with the original outer residual path."""

    def __init__(
        self,
        dim: int,
        reduction: int = 16,
        nhead: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        use_mgf: bool = True,
    ) -> None:
        super().__init__()
        self.alme = ALME(dim, reduction=reduction)
        self.attn = ConditionedSelfAttention(
            dim=dim,
            nhead=nhead,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_mgf=use_mgf,
        )

    def forward(
        self,
        x: torch.Tensor,
        gate_state: GateState,
        permutation: torch.Tensor | None = None,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        f_hat, descriptor = self.alme(x)
        csa_output = self.attn(
            f_hat,
            descriptor,
            gate_state=gate_state,
            permutation=permutation,
            return_gates=return_gates,
        )

        if return_gates:
            output, gates = csa_output
            return x + output, gates
        return x + csa_output


class IDENGATE(nn.Module):
    """Three-stage IDENGATE classifier.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        depth: int = 3,
        base_dim: int = 64,
        reduction: int = 16,
        nhead: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        alpha: float = 0.1,
        use_mgf: bool = True,
    ) -> None:
        super().__init__()
        if depth not in (1, 2, 3):
            raise ValueError("depth must be one of {1,2,3}")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.depth = int(depth)
        self.alpha = float(alpha)
        self.use_mgf = bool(use_mgf)

        channels: Sequence[int] = (
            base_dim,
            base_dim * 2,
            base_dim * 4,
            base_dim * 8,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        downsample: list[nn.Module] = []
        for stage_index in range(depth):
            dim = channels[stage_index]
            blocks.append(
                IDENGATEBlock(
                    dim=dim,
                    reduction=reduction,
                    nhead=nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_mgf=use_mgf,
                )
            )
            downsample.append(
                nn.Sequential(
                    nn.Conv2d(dim, channels[stage_index + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(channels[stage_index + 1]),
                    nn.ReLU(inplace=True),
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.downsample = nn.ModuleList(downsample)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[depth], num_classes)

    def _state(
        self,
        alpha: float | None,
        identity_roles: Iterable[str] | None,
        shuffle_gates: bool,
    ) -> GateState:
        return GateState.from_values(
            alpha=self.alpha if alpha is None else alpha,
            identity_roles=identity_roles,
            shuffle=shuffle_gates,
        )

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
        state = self._state(alpha, identity_roles, shuffle_gates)
        if state.shuffle and gate_permutation is None:
            gate_permutation = torch.randperm(x.shape[0], device=x.device)
        x = self.stem(x)
        all_gates: list[Mapping[str, torch.Tensor]] = []

        for block, downsample in zip(self.blocks, self.downsample):
            if return_gates:
                x, gates = block(
                    x,
                    gate_state=state,
                    permutation=gate_permutation,
                    return_gates=True,
                )
                all_gates.append(gates)
            else:
                x = block(
                    x,
                    gate_state=state,
                    permutation=gate_permutation,
                    return_gates=False,
                )
            x = downsample(x)

        logits = self.fc(self.gap(x).flatten(1))
        if return_gates:
            return logits, all_gates
        return logits

    @torch.no_grad()
    def forward_mgf_on(self, x: torch.Tensor) -> torch.Tensor:
        """Primary learned-gate state (alpha=0.1 by default)."""

        return self(x, alpha=self.alpha)

    @torch.no_grad()
    def forward_mgf_off(self, x: torch.Tensor) -> torch.Tensor:
        """Exact same-checkpoint identity-gate state (alpha=0)."""

        return self(x, alpha=0.0)

    @torch.no_grad()
    def forward_identity_intervention(
        self,
        x: torch.Tensor,
        roles: Iterable[str],
    ) -> torch.Tensor:
        """Set selected role gates to identity in the same checkpoint."""

        return self(x, alpha=self.alpha, identity_roles=roles)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_primary_model(num_classes: int = 5, in_channels: int = 3) -> IDENGATE:
    """Build the primary three-block IDENGATE configuration."""

    return IDENGATE(
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
    )


# Backward-compatible names for early internal scripts.
CMABlock = IDENGATEBlock
CMANet = IDENGATE


if __name__ == "__main__":
    primary = build_primary_model()
    no_mgf = IDENGATE(use_mgf=False)
    print(f"IDENGATE parameters: {count_trainable_parameters(primary):,}")
    print(f"no-MGF parameters:   {count_trainable_parameters(no_mgf):,}")
