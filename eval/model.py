import torch
import torch.nn as nn


class ALME(nn.Module):

    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        z = self.dwconv(x)
        z = self.bn(z)
        z = z * self.se(z)
        z = self.proj(z)
        return z


class ConditionedSelfAttention(nn.Module):

    def __init__(self, dim, nhead=4, mlp_ratio=4, dropout=0.1,
                 use_conditioned=True, alpha=0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * mlp_ratio,
            dropout=dropout
        )
        self.use_conditioned = use_conditioned
        self.alpha = alpha

        self.gq = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)
        self.gk = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)
        self.gv = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)

        for m in (self.gq, self.gk, self.gv):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.view(b, c, h * w).permute(2, 0, 1)  # [HW, B, C]

        if not self.use_conditioned:
            out = self.encoder(seq)
            return out.permute(1, 2, 0).view(b, c, h, w)

        pooled = x.mean(dim=(2, 3), keepdim=True)
        phi_q = 1.0 + self.alpha * torch.tanh(self.gq(pooled)).view(b, c)
        phi_k = 1.0 + self.alpha * torch.tanh(self.gk(pooled)).view(b, c)
        phi_v = 1.0 + self.alpha * torch.tanh(self.gv(pooled)).view(b, c)

        phi_q, phi_k, phi_v = phi_q.unsqueeze(0), phi_k.unsqueeze(0), phi_v.unsqueeze(0)

        enc = self.encoder
        q, k, v = seq * phi_q, seq * phi_k, seq * phi_v

        attn_out, _ = enc.self_attn(q, k, v, need_weights=False)
        src = enc.norm1(seq + enc.dropout1(attn_out))

        ffn_out = enc.linear2(enc.dropout(enc.activation(enc.linear1(src))))
        src = enc.norm2(src + enc.dropout2(ffn_out))

        return src.permute(1, 2, 0).view(b, c, h, w)


class CMABlock(nn.Module):

    def __init__(self, dim, reduction=16):
        super().__init__()
        self.alme = ALME(dim, reduction)
        self.attn = ConditionedSelfAttention(dim)

    def forward(self, x):
        return x + self.attn(self.alme(x))


class CMANet(nn.Module):

    def __init__(self, num_classes=5, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            CMABlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.stage2 = nn.Sequential(
            CMABlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.stage3 = nn.Sequential(
            CMABlock(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
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
