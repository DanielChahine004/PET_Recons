# cyl_petnet_3d_unified.py
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(c: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class CircularPadConv3d(nn.Module):
    """Conv3d with circular pad on W (azimuth), zero on T/H."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (1, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        dilation: Tuple[int, int, int] = (1, 1, 1),
        bias: bool = False,
    ):
        super().__init__()
        self.kt, self.kh, self.kw = kernel_size
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dil_t, dil_h, dil_w = self.conv.dilation
        pad_t = ((self.kt - 1) // 2) * dil_t
        pad_h = ((self.kh - 1) // 2) * dil_h
        pad_w = ((self.kw - 1) // 2) * dil_w
        if pad_w > 0:
            x = F.pad(x, (pad_w, pad_w, 0, 0, 0, 0), mode="circular")
        if pad_h > 0 or pad_t > 0:
            x = F.pad(x, (0, 0, pad_h, pad_h, pad_t, pad_t), mode="constant", value=0.0)
        return self.conv(x)


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


class SpatialAttention3D(nn.Module):
    """CBAM spatial attention in H×W, preserving T."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, 7, 7), padding=(0, 3, 3), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(s))
        return x * attn


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.conv1 = CircularPadConv3d(in_ch, out_ch, kernel_size=(1, 3, 3), stride=stride, bias=False)
        self.gn1 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU()
        self.conv2 = CircularPadConv3d(out_ch, out_ch, kernel_size=(1, 3, 3), stride=(1, 1, 1), bias=False)
        self.gn2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.se = SEBlock3D(out_ch, reduction=8)
        self.proj = None
        if in_ch != out_ch or stride != (1, 1, 1):
            self.proj = nn.Sequential(
                CircularPadConv3d(in_ch, out_ch, kernel_size=(1, 1, 1), stride=stride, bias=False),
                nn.GroupNorm(_gn_groups(out_ch), out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.gn1(self.conv1(x)))
        y = self.se(self.gn2(self.conv2(y)))
        if self.proj is not None:
            x = self.proj(x)
        return self.act(y + x)


class _Branch3D(nn.Module):
    """One branch (inner or outer). Downsample H,W; keep T until global pool."""
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            CircularPadConv3d(in_ch, base, kernel_size=(1, 3, 3), stride=(1, 1, 1), bias=False),
            nn.GroupNorm(_gn_groups(base), base),
            nn.SiLU(),
            ResBlock3D(base, base, stride=(1, 1, 1)),
        )
        # Downsample spatially only: stride (T,H,W) = (1,2,2)
        self.stage2 = ResBlock3D(base, base * 2, stride=(1, 2, 2))
        self.stage3 = ResBlock3D(base * 2, base * 4, stride=(1, 2, 2))
        self.stage4 = ResBlock3D(base * 4, base * 4, stride=(1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class DualBranchGate(nn.Module):
    """Energy-aware fusion on (T,H,W)."""
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, f_in: torch.Tensor, f_out: torch.Tensor, e_in: torch.Tensor, e_out: torch.Tensor):
        gin = torch.mean(f_in, dim=(2, 3, 4))   # (B,C)
        gout = torch.mean(f_out, dim=(2, 3, 4)) # (B,C)
        g = torch.cat([gin, gout, e_in, e_out], dim=1)  # (B, 2C+2)
        w = self.softmax(self.mlp(g))                   # (B,2)
        wi = w[:, 0].view(-1, 1, 1, 1, 1)
        wo = w[:, 1].view(-1, 1, 1, 1, 1)
        fused = wi * f_in + wo * f_out
        return fused, w


class CylindricalPetNet3D(nn.Module):
    """
    Unified-input version.
    Accepts x: (B, C, T, H, W)
      C = 2           -> [inner, outer]
      C = 4           -> [inner, outer, sinθ, cosθ]
      C = 6           -> [inner, outer, sinθ, cosθ, r_in_norm, r_out_norm]
    Extras (if present) are duplicated to each branch input.
    """
    def __init__(self, output_features: int = 6, extra_channels: int = 0, base_channels: int = 32, post_channels: int = 256, dropout_p: float = 0.2):
        super().__init__()
        assert extra_channels in (0, 2, 4), "extra_channels must be 0, 2, or 4"
        self.extra_channels = extra_channels

        in_ch_branch = 1 + self.extra_channels  # each branch sees its signal + shared extras

        self.inner_branch = _Branch3D(in_ch=in_ch_branch, base=base_channels)
        self.outer_branch = _Branch3D(in_ch=in_ch_branch, base=base_channels)

        feat_ch = base_channels * 4
        self.fuse_gate = DualBranchGate(channels=feat_ch, hidden=64)
        self.spatial_attn = SpatialAttention3D()

        self.post = nn.Sequential(
            ResBlock3D(feat_ch, 192, stride=(1, 2, 2)),
            ResBlock3D(192, post_channels, stride=(1, 1, 1)),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.head = nn.Sequential(
            nn.Linear(post_channels, 128),
            nn.SiLU(),
            nn.Linear(128, output_features),
        )

    @staticmethod
    def _branch_energy(signal_1ch: torch.Tensor) -> torch.Tensor:
        # signal_1ch: (B,1,T,H,W)
        e = torch.sum(torch.relu(signal_1ch), dim=(2, 3, 4), keepdim=False)
        return torch.log1p(e)  # (B,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W) with C = 2 + extra_channels
        returns: (B, 3) (x, y, z)
        """
        assert x.dim() == 5, "Expect (B,C,T,H,W)"
        B, C, T, H, W = x.shape
        assert C == 2 + self.extra_channels, f"Expected C={2 + self.extra_channels}, got {C}"

        inner_sig = x[:, 0:1, ...]  # (B,1,T,H,W)
        outer_sig = x[:, 1:2, ...]  # (B,1,T,H,W)
        extras = x[:, 2:, ...] if self.extra_channels > 0 else None  # (B,E,T,H,W) or None

        inner_in = torch.cat([inner_sig, extras], dim=1) if extras is not None else inner_sig
        outer_in = torch.cat([outer_sig, extras], dim=1) if extras is not None else outer_sig

        e_in = self._branch_energy(inner_sig)
        e_out = self._branch_energy(outer_sig)

        f_in = self.inner_branch(inner_in)
        f_out = self.outer_branch(outer_in)

        fused, _ = self.fuse_gate(f_in, f_out, e_in, e_out)
        fused = self.spatial_attn(fused)

        y = self.post(fused)
        y = self.pool(y).view(B, -1)
        y = self.dropout(y)
        xyz = self.head(y)
        return xyz


# ---- self-test ----
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example: C=6 -> [inner, outer, sinθ, cosθ, r_in, r_out]
    B, C, T, H, W = 2, 6, 3, 62, 519
    x = torch.rand(B, C, T, H, W, device=device)

    model = CylindricalPetNet3D(extra_channels=4, base_channels=32, post_channels=256).to(device)
    y = model(x)
    print("Output:", y.shape, y[0])