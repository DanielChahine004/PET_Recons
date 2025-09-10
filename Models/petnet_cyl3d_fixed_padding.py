import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Literal


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * (mask / keep)


def make_norm(norm: Literal["group", "batch", "instance"], num_channels: int, groups: int = 8):
    if norm == "group":
        g = max(1, min(groups, num_channels))
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    elif norm == "batch":
        return nn.BatchNorm3d(num_channels)
    else:
        return nn.InstanceNorm3d(num_channels, affine=True)


# -----------------------------------------------------------
# Circular-padded Conv3d (wrap-around on H only)
# -----------------------------------------------------------
class Conv3dCircH(nn.Module):
    """
    Conv3d with **circular** padding on the HEIGHT dimension (H),
    and normal zero-padding on T and W. Works for kernel_size 1 or 3 cleanly.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=(1, 1, 1), bias=False):
        super().__init__()
        if isinstance(kernel_size, int):
            kD = kH = kW = kernel_size
        else:
            kD, kH, kW = kernel_size
        # We pre-pad H ourselves; so we set conv padding=(padT, padH=0, padW)
        padT = (kD - 1) // 2
        padW = (kW - 1) // 2
        self.pad_h = (kH - 1) // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(kD, kH, kW),
                              stride=stride, padding=(padT, 0, padW), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_h > 0:
            # F.pad pads in reverse order: (Wl, Wr, Hl, Hr, Dl, Dr)
            # For circular padding on H: (0, 0, pad_h, pad_h, 0, 0)
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h, 0, 0), mode="circular")
        return self.conv(x)


# -----------------------------------------------------------
# Enhanced Residual Block with more regularization
# -----------------------------------------------------------
class ResidualBlock3D(nn.Module):
    """
    Enhanced residual block with better regularization for generalization.
    Uses circular padding on Height dimension.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: Tuple[int, int, int] = (1, 2, 2),
            norm: Literal["group", "batch", "instance"] = "group",
            drop_path: float = 0.0,
            spatial_dropout: float = 0.1,
            groups_for_gn: int = 8,
    ):
        super().__init__()
        self.conv1 = Conv3dCircH(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self.n1 = make_norm(norm, out_channels, groups_for_gn)
        self.act = nn.GELU()

        # Add spatial dropout between convolutions
        self.spatial_dropout = nn.Dropout3d(spatial_dropout) if spatial_dropout > 0 else nn.Identity()

        self.conv2 = Conv3dCircH(out_channels, out_channels, kernel_size=3, stride=(1, 1, 1), bias=False)
        self.n2 = make_norm(norm, out_channels, groups_for_gn)

        self.shortcut = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_norm(norm, out_channels, groups_for_gn),
            )
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.n1(self.conv1(x)))
        out = self.spatial_dropout(out)  # Add spatial dropout
        out = self.n2(self.conv2(out))

        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = self.act(identity + self.drop_path(out))
        return out


# -----------------------------------------------------------
# Improved PetNetCyl3D with better regularization
# -----------------------------------------------------------
class PetNetCyl3D(nn.Module):
    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 6,
            base_channels: int = 8,  # Reduced from 16 to 8
            norm: Literal["group", "batch", "instance"] = "group",
            groups_for_gn: int = 4,  # Reduced groups
            dropout3d: float = 0.15,  # Increased dropout
            spatial_dropout: float = 0.1,  # New spatial dropout
            drop_path_rate: float = 0.15,  # Increased drop path
            fc_dropout: float = 0.6,  # Increased FC dropout
            num_layers: int = 4,  # Reduced from 5 to 4 layers
    ):
        super().__init__()
        print(f"Loading PetNetCyl3D (regularized, circular H) - base_channels={base_channels}, layers={num_layers}")

        self.act = nn.GELU()

        # Simpler stem
        self.stem = nn.Sequential(
            Conv3dCircH(in_channels, base_channels, kernel_size=3, stride=(1, 1, 1), bias=False),
            make_norm(norm, base_channels, groups_for_gn),
            nn.GELU(),
            nn.Dropout3d(dropout3d) if dropout3d > 0 else nn.Identity(),
        )

        # Reduced channel progression: 8 -> 16 -> 32 -> 64 -> 128 (instead of -> 256)
        chs = [base_channels * (2 ** i) for i in range(num_layers + 1)]
        strides = [(1, 2, 2)] * num_layers
        dprs = [drop_path_rate * i / (num_layers - 1) for i in range(num_layers)]

        # Create layers dynamically
        layers = []
        for i in range(num_layers):
            layers.append(ResidualBlock3D(
                chs[i], chs[i + 1],
                stride=strides[i],
                norm=norm,
                drop_path=dprs[i],
                spatial_dropout=spatial_dropout,
                groups_for_gn=groups_for_gn
            ))

        self.backbone = nn.Sequential(*layers)

        # Global average pool over (T,H,W)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Simplified and more regularized head
        hidden_dim = max(64, chs[-1] // 2)  # Adaptive hidden dimension

        self.fc_shared = nn.Sequential(
            nn.Linear(chs[-1], hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=True),  # Additional layer for better representation
            nn.GELU(),
            nn.Dropout(fc_dropout * 0.5),  # Reduced dropout for final layer
        )

        # Separate heads for inner and outer endpoints
        head_input_dim = hidden_dim // 2
        self.head_inner = nn.Sequential(
            nn.Linear(head_input_dim, 16, bias=True),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(16, out_channels//2, bias=True)
        )

        self.head_outer = nn.Sequential(
            nn.Linear(head_input_dim, 16, bias=True),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(16, out_channels//2, bias=True)
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C=2, T, H, W); H is the unrolled circumference (we circular-pad it).
        returns: (B, 6) = [cosφ1, sinφ1, z1, cosφ2, sinφ2, z2]
        """
        x = self.stem(x)
        x = self.backbone(x)
        x = self.gap(x).flatten(1)
        x = self.fc_shared(x)

        e1 = self.head_inner(x)  # (B,4) - changed to 4 outputs
        e2 = self.head_outer(x)  # (B,4) - changed to 4 outputs

        # Encourage valid cosine/sine by soft-normalizing
        def normalize_cos_sin(v: torch.Tensor) -> torch.Tensor:
            cos_sin = v[..., :2]
            z = v[..., 2:3]
            extra = v[..., 3:] if v.shape[-1] > 3 else None
            
            # More stable normalization
            cos_sin = torch.tanh(cos_sin * 0.5)  # Reduced scaling for stability
            norm = torch.clamp(torch.linalg.norm(cos_sin, dim=-1, keepdim=True), min=1e-6)
            cos_sin = cos_sin / norm
            
            if extra is not None:
                return torch.cat([cos_sin, z, extra], dim=-1)
            else:
                return torch.cat([cos_sin, z], dim=-1)

        e1 = normalize_cos_sin(e1)
        e2 = normalize_cos_sin(e2)
        out = torch.cat([e1, e2], dim=-1)  # (B, 8) if 4 outputs each
        return out

    def _init_weights(self):
        """More conservative weight initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                # Use smaller initialization for better generalization
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu", mode="fan_out")
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# -----------------------------------------------------------
# Compact version for small datasets
# -----------------------------------------------------------
class PetNetCyl3DCompact(nn.Module):
    """Ultra-compact version for very small datasets."""

    def __init__(
            self,
            in_channels: int = 4,
            base_channels: int = 6,  # Very small base
            dropout_rate: float = 0.7,  # Aggressive dropout
            out_features: int = 6,
    ):
        super().__init__()
        print("Loading PetNetCyl3D Compact (minimal overfitting, circular H)")

        self.backbone = nn.Sequential(
            Conv3dCircH(in_channels, base_channels, 3, (1, 1, 1)),
            nn.GroupNorm(2, base_channels),
            nn.GELU(),
            nn.Dropout3d(0.2),

            Conv3dCircH(base_channels, base_channels * 2, 3, (1, 2, 2)),
            nn.GroupNorm(2, base_channels * 2),
            nn.GELU(),
            nn.Dropout3d(0.3),

            Conv3dCircH(base_channels * 2, base_channels * 4, 3, (1, 2, 2)),
            nn.GroupNorm(4, base_channels * 4),
            nn.GELU(),
            nn.Dropout3d(0.4),
        )

        self.gap = nn.AdaptiveAvgPool3d(1)

        self.head = nn.Sequential(
            nn.Linear(base_channels * 4, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, out_features)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x).flatten(1)
        return self.head(x)


# -----------------------------------------------------------
# Compact version without GAP (uses full feature map)
# -----------------------------------------------------------
class PetNetCyl3DFullFeatures(nn.Module):
    """Compact version that uses ALL output features (no GAP), TorchScript compatible."""

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 6,
        input_shape=(2, 3, 207, 41),  # (C, T, H, W) so we can compute flatten_dim
        dropout_rate: float = 0.5,
        out_features: int = 6,
    ):
        super().__init__()
        print("Loading PetNetCyl3D (Full Features TorchScript Compatible, circular H)")

        # Backbone with stride-based downsampling in H,W
        # Note: Using Conv3dCircH for consistency with circular padding
        self.conv1 = Conv3dCircH(in_channels, base_channels, kernel_size=3, stride=(1, 1, 1))
        self.norm1 = nn.GroupNorm(2, base_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout3d(0.2)

        self.conv2 = Conv3dCircH(base_channels, base_channels * 2, kernel_size=3, stride=(1, 2, 2))
        self.norm2 = nn.GroupNorm(2, base_channels * 2)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout3d(0.3)

        self.conv3 = Conv3dCircH(base_channels * 2, base_channels * 4, kernel_size=3, stride=(1, 2, 2))
        self.norm3 = nn.GroupNorm(4, base_channels * 4)
        self.act3 = nn.GELU()
        self.drop3 = nn.Dropout3d(0.4)

        # Precompute flatten_dim from input shape
        flatten_dim = self._compute_flatten_dim(in_channels, input_shape)
        print(f"[Init] Flattened feature size = {flatten_dim}")

        # Fully connected head
        self.head = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, out_features)
        )

    def _compute_flatten_dim(self, in_channels, input_shape):
        """Compute feature map flatten size at init time (TorchScript safe)."""
        C, T, H, W = input_shape
        dummy = torch.zeros(1, in_channels, T, H, W)
        with torch.no_grad():
            # Forward through backbone manually to get output shape
            x = self.act1(self.norm1(self.conv1(dummy)))
            x = self.drop1(x)
            x = self.act2(self.norm2(self.conv2(x)))
            x = self.drop2(x)
            x = self.act3(self.norm3(self.conv3(x)))
            x = self.drop3(x)
        return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through backbone manually
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.act3(self.norm3(self.conv3(x)))
        x = self.drop3(x)
        
        # Flatten and pass through head
        x = torch.flatten(x, 1)
        return self.head(x)


# -----------------------------------------------------------
# Test function
# -----------------------------------------------------------
if __name__ == "__main__":
    # Test the three core models with circular H padding
    B, C, T, H, W = 4, 2, 3, 207, 41

    print("Testing core models with circular padding on HEIGHT dimension:")
    print("=" * 60)

    # Test main model
    model = PetNetCyl3D(in_channels=C, base_channels=8, num_layers=3)
    x = torch.randn(B, C, T, H, W)
    y_pred = model(x)
    print(f"Regular model output shape: {y_pred.shape}")

    # Test compact model
    compact_model = PetNetCyl3DCompact(in_channels=C, base_channels=6, out_features=6)
    y_pred_compact = compact_model(x)
    print(f"Compact model output shape: {y_pred_compact.shape}")

    # Test full features model
    full_model = PetNetCyl3DFullFeatures(in_channels=C, base_channels=6, 
                                         input_shape=(C, T, H, W), out_features=6)
    y_pred_full = full_model(x)
    print(f"Full features model output shape: {y_pred_full.shape}")

    # Count parameters
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nParameter counts:")
    print("-" * 30)
    print(f"Regular model parameters: {count_params(model):,}")
    print(f"Compact model parameters: {count_params(compact_model):,}")
    print(f"Full features model parameters: {count_params(full_model):,}")

    print("\nAll three core models successfully use circular padding on HEIGHT dimension!")