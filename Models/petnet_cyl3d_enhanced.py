# Modified version with positional features added to input
# takes attention_petnet.py and places the global attention layer after layer 1
# for higher dimensional G.A. Employs point-wise seperation convolutions over
# full fat classical convolutions for an 8-10x reduction in paramater count. 
# Also introduces a residual connection after the global attention, and layer norm
# for the global attention (Apparently batch norm works better for Convolutions, 
# and layer norm works better for Transformers...we'll see about that). 
# Uses 3 full connected layers with dropout for a more regression head.   
# Modified to use windowed attention with 4x4 non-overlapping windows.
# ADDED: Positional features (height and width indices) to input channels

import torch
import torch.nn as nn

###############################################################################
# Positional Feature Generator
###############################################################################


# petnet_cyl3d.py
from typing import Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Circular-padded Conv3d (wrap-around on W only)
# -----------------------------------------------------------
class Conv3dCircW(nn.Module):
    """
    Conv3d with **circular** padding on the last spatial dim (W = circumference),
    and normal zero-padding on T and H. Works for kernel_size 1 or 3 cleanly.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=(1, 1, 1), bias=False):
        super().__init__()
        if isinstance(kernel_size, int):
            kD = kH = kW = kernel_size
        else:
            kD, kH, kW = kernel_size
        # We pre-pad W ourselves; so we set conv padding=(padT, padH, padW=0)
        padT = (kD - 1) // 2
        padH = (kH - 1) // 2
        self.pad_w = (kW - 1) // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(kD, kH, kW),
                              stride=stride, padding=(padT, padH, 0), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_w > 0:
            # F.pad pads in reverse order: (Wl, Wr, Hl, Hr, Dl, Dr)
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0, 0, 0), mode="circular")
        return self.conv(x)


# -----------------------------------------------------------
# Enhanced Residual Block with more regularization
# -----------------------------------------------------------
class ResidualBlock3D(nn.Module):
    """
    Enhanced residual block with better regularization for generalization.
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
        self.conv1 = Conv3dCircW(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self.n1 = make_norm(norm, out_channels, groups_for_gn)
        self.act = nn.GELU()

        # Add spatial dropout between convolutions
        self.spatial_dropout = nn.Dropout3d(spatial_dropout) if spatial_dropout > 0 else nn.Identity()

        self.conv2 = Conv3dCircW(out_channels, out_channels, kernel_size=3, stride=(1, 1, 1), bias=False)
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



# --------------------------------------------------------------------------------
# Sqeeze Excitation (SE) Block and Enhanced Residual Block with SE and Multi-Scale
# --------------------------------------------------------------------------------
class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, max(channels // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(channels // reduction, 1), channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class EnhancedResidualBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: Tuple[int, int, int] = (1, 2, 2),
            norm: Literal["group", "batch", "instance"] = "group",
            drop_path: float = 0.0,
            spatial_dropout: float = 0.1,
            groups_for_gn: int = 8,
            se_reduction: int = 16,
            multi_scale: bool = True,
    ):
        super().__init__()
        
        self.n1 = make_norm(norm, in_channels, groups_for_gn)
        self.act1 = nn.GELU()
        
        self.conv1 = Conv3dCircW(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        
        self.n2 = make_norm(norm, out_channels, groups_for_gn)
        self.act2 = nn.GELU()
        
        self.spatial_dropout = nn.Dropout3d(spatial_dropout) if spatial_dropout > 0 else nn.Identity()
        
        if multi_scale:
            mid_channels = out_channels // 2
            self.conv2a = Conv3dCircW(out_channels, mid_channels, kernel_size=1, stride=(1, 1, 1), bias=False)
            self.conv2b = Conv3dCircW(out_channels, out_channels - mid_channels, kernel_size=3, stride=(1, 1, 1), bias=False)
            self.multi_scale = True
        else:
            self.conv2 = Conv3dCircW(out_channels, out_channels, kernel_size=3, stride=(1, 1, 1), bias=False)
            self.multi_scale = False
        
        self.se = SEBlock3D(out_channels, se_reduction)
        
        self.shortcut = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                make_norm(norm, in_channels, groups_for_gn),
                nn.GELU(),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )
        
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        if self.multi_scale:
            nn.init.zeros_(self.conv2b.conv.weight)
        else:
            nn.init.zeros_(self.conv2.conv.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.act1(self.n1(x))
        out = self.conv1(out)
        
        out = self.act2(self.n2(out))
        out = self.spatial_dropout(out)
        
        if self.multi_scale:
            out_1x1 = self.conv2a(out)
            out_3x3 = self.conv2b(out)
            out = torch.cat([out_1x1, out_3x3], dim=1)
        else:
            out = self.conv2(out)
        
        out = self.se(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        out = identity + self.drop_path(out)
        
        return out

# -----------------------------------------------------------
# PetNetCyl3DEnhanced
# -----------------------------------------------------------
class PetNetCyl3DFullFeaturesEnhanced(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 6,
        input_shape=(2, 3, 207, 41),
        dropout_rate: float = 0.5,
        out_features: int = 6,
        max_drop_path: float = 0.2,
        se_reduction: int = 16,
        multi_scale: bool = True,
        norm_type: Literal["group", "batch", "instance"] = "group",
    ):
        super().__init__()
        print("Loading PetNetCyl3D Enhanced (Full Features with Residual Blocks)")

        self.stem = nn.Sequential(
            Conv3dCircW(in_channels, base_channels, kernel_size=3, stride=(1, 1, 1), bias=False),
            make_norm(norm_type, base_channels),
            nn.GELU(),
            nn.Dropout3d(0.1),
        )

        total_blocks = 6

        self.stage1 = nn.Sequential(
            EnhancedResidualBlock3D(
                in_channels=base_channels, 
                out_channels=base_channels * 2,
                stride=(1, 1, 1),
                norm=norm_type, 
                drop_path=max_drop_path * (0 / max(total_blocks - 1, 1)),
                spatial_dropout=0.15,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            ),
            EnhancedResidualBlock3D(
                in_channels=base_channels * 2, 
                out_channels=base_channels * 2,
                stride=(1, 2, 2),
                norm=norm_type, 
                drop_path=max_drop_path * (1 / max(total_blocks - 1, 1)),
                spatial_dropout=0.2,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            )
        )

        self.stage2 = nn.Sequential(
            EnhancedResidualBlock3D(
                in_channels=base_channels * 2, 
                out_channels=base_channels * 4,
                stride=(1, 1, 1),
                norm=norm_type, 
                drop_path=max_drop_path * (2 / max(total_blocks - 1, 1)),
                spatial_dropout=0.25,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            ),
            EnhancedResidualBlock3D(
                in_channels=base_channels * 4, 
                out_channels=base_channels * 4,
                stride=(1, 2, 2),
                norm=norm_type, 
                drop_path=max_drop_path * (3 / max(total_blocks - 1, 1)),
                spatial_dropout=0.3,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            )
        )

        self.stage3 = nn.Sequential(
            EnhancedResidualBlock3D(
                in_channels=base_channels * 4, 
                out_channels=base_channels * 8,
                stride=(1, 1, 1),
                norm=norm_type, 
                drop_path=max_drop_path * (4 / max(total_blocks - 1, 1)),
                spatial_dropout=0.35,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            ),
            EnhancedResidualBlock3D(
                in_channels=base_channels * 8, 
                out_channels=base_channels * 8,
                stride=(1, 2, 2),
                norm=norm_type, 
                drop_path=max_drop_path * (5 / max(total_blocks - 1, 1)),
                spatial_dropout=0.4,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            )
        )

        self.backbone = nn.Sequential(
            self.stem,
            self.stage1,
            self.stage2,
            self.stage3
        )

        flatten_dim = self._compute_flatten_dim(in_channels, input_shape)
        print(f"[Init] Flattened feature size = {flatten_dim}")

        self.head = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(128, out_features)
        )

        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def _compute_flatten_dim(self, in_channels, input_shape):
        C, T, H, W = input_shape
        dummy = torch.zeros(1, C, T, H, W)
        with torch.no_grad():
            out = self.backbone(dummy)
        return out.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        features = {}
        x = self.stem(x)
        features['stem'] = x
        
        x = self.stage1(x)
        features['stage1'] = x
        
        x = self.stage2(x)
        features['stage2'] = x
        
        x = self.stage3(x)
        features['stage3'] = x
        
        return features


class SplitResidualBlock3D(EnhancedResidualBlock3D):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        self.out_channels = out_channels
        
        assert self.out_channels % 2 == 0, f"out_channels ({out_channels}) must be even for +/- split"
        self.split_point = self.out_channels // 2
        
        super().__init__(in_channels, out_channels, *args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.act1(self.n1(x))
        out = self.conv1(out)
        out = self.act2(self.n2(out))
        out = self.spatial_dropout(out)
        
        if self.multi_scale:
            out_1x1 = self.conv2a(out)
            out_3x3 = self.conv2b(out)
            out = torch.cat([out_1x1, out_3x3], dim=1)
        else:
            out = self.conv2(out)
        
        out = self.se(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        out = self.drop_path(out)
        
        out_add = out[:, :self.split_point]
        out_sub = out[:, self.split_point:]
        
        identity_add = identity[:, :self.split_point]
        identity_sub = identity[:, self.split_point:]
        
        result_add = identity_add + out_add
        result_sub = identity_sub - out_sub
        
        return torch.cat([result_add, result_sub], dim=1)


# -----------------------------------------------------------
# SplitPetNetCyl3DEnhanced
# -----------------------------------------------------------
class SplitPetNetCyl3DFullFeaturesEnhanced(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 6,
        input_shape=(2, 3, 207, 41),
        dropout_rate: float = 0.5,
        out_features: int = 6,
        max_drop_path: float = 0.2,
        se_reduction: int = 16,
        multi_scale: bool = True,
        norm_type: Literal["group", "batch", "instance"] = "group",
    ):
        super().__init__()
        print("Loading PetNetCyl3D Enhanced (Full Features with Residual Blocks)")

        self.stem = nn.Sequential(
            Conv3dCircW(in_channels, base_channels, kernel_size=3, stride=(1, 1, 1), bias=False),
            make_norm(norm_type, base_channels),
            nn.GELU(),
            nn.Dropout3d(0.1),
        )

        total_blocks = 6

        self.stage1 = nn.Sequential(
            SplitResidualBlock3D(
                in_channels=base_channels, 
                out_channels=base_channels * 2,
                stride=(1, 1, 1),
                norm=norm_type, 
                drop_path=max_drop_path * (0 / max(total_blocks - 1, 1)),
                spatial_dropout=0.15,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            ),
            SplitResidualBlock3D(
                in_channels=base_channels * 2, 
                out_channels=base_channels * 2,
                stride=(1, 2, 2),
                norm=norm_type, 
                drop_path=max_drop_path * (1 / max(total_blocks - 1, 1)),
                spatial_dropout=0.2,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            )
        )

        self.stage2 = nn.Sequential(
            SplitResidualBlock3D(
                in_channels=base_channels * 2, 
                out_channels=base_channels * 4,
                stride=(1, 1, 1),
                norm=norm_type, 
                drop_path=max_drop_path * (2 / max(total_blocks - 1, 1)),
                spatial_dropout=0.25,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            ),
            SplitResidualBlock3D(
                in_channels=base_channels * 4, 
                out_channels=base_channels * 4,
                stride=(1, 2, 2),
                norm=norm_type, 
                drop_path=max_drop_path * (3 / max(total_blocks - 1, 1)),
                spatial_dropout=0.3,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            )
        )

        self.stage3 = nn.Sequential(
            SplitResidualBlock3D(
                in_channels=base_channels * 4, 
                out_channels=base_channels * 8,
                stride=(1, 1, 1),
                norm=norm_type, 
                drop_path=max_drop_path * (4 / max(total_blocks - 1, 1)),
                spatial_dropout=0.35,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            ),
            SplitResidualBlock3D(
                in_channels=base_channels * 8, 
                out_channels=base_channels * 8,
                stride=(1, 2, 2),
                norm=norm_type, 
                drop_path=max_drop_path * (5 / max(total_blocks - 1, 1)),
                spatial_dropout=0.4,
                se_reduction=se_reduction, 
                multi_scale=multi_scale
            )
        )

        self.backbone = nn.Sequential(
            self.stem,
            self.stage1,
            self.stage2,
            self.stage3
        )

        flatten_dim = self._compute_flatten_dim(in_channels, input_shape)
        print(f"[Init] Flattened feature size = {flatten_dim}")

        self.head = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(128, out_features)
        )

        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def _compute_flatten_dim(self, in_channels, input_shape):
        C, T, H, W = input_shape
        dummy = torch.zeros(1, C, T, H, W)
        with torch.no_grad():
            out = self.backbone(dummy)
        return out.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        features = {}
        x = self.stem(x)
        features['stem'] = x
        
        x = self.stage1(x)
        features['stage1'] = x
        
        x = self.stage2(x)
        features['stage2'] = x
        
        x = self.stage3(x)
        features['stage3'] = x
        
        return features


# -----------------------------------------------------------
# Improved PetNetCyl3D with better regularization
# -----------------------------------------------------------
class PetNetCyl3D(nn.Module):
    def __init__(
            self,
            in_channels: int = 4,
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
        print(f"Loading PetNetCyl3D (regularized) - base_channels={base_channels}, layers={num_layers}")

        self.act = nn.GELU()

        # Simpler stem
        self.stem = nn.Sequential(
            Conv3dCircW(in_channels, base_channels, kernel_size=3, stride=(1, 1, 1), bias=False),
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
            nn.Linear(16, 4, bias=True)
        )

        self.head_outer = nn.Sequential(
            nn.Linear(head_input_dim, 16, bias=True),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(16, 4, bias=True)
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C=2, T, H, W); W is the unrolled circumference (we circular-pad it).
        returns: (B, 6) = [cosφ1, sinφ1, z1, cosφ2, sinφ2, z2]
        """
        x = self.stem(x)
        x = self.backbone(x)
        x = self.gap(x).flatten(1)
        x = self.fc_shared(x)

        e1 = self.head_inner(x)  # (B,3)
        e2 = self.head_outer(x)  # (B,3)

        # Encourage valid cosine/sine by soft-normalizing
        def normalize_cos_sin(v: torch.Tensor) -> torch.Tensor:
            cos_sin = v[..., :2]
            z = v[..., 2:3]
            # More stable normalization
            cos_sin = torch.tanh(cos_sin * 0.5)  # Reduced scaling for stability
            norm = torch.clamp(torch.linalg.norm(cos_sin, dim=-1, keepdim=True), min=1e-6)
            cos_sin = cos_sin / norm
            return torch.cat([cos_sin, z], dim=-1)

        e1 = normalize_cos_sin(e1)
        e2 = normalize_cos_sin(e2)
        out = torch.cat([e1, e2], dim=-1)  # (B,6)
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
# Alternative: Even more compact version for small datasets
# -----------------------------------------------------------
class PetNetCyl3DCompact(nn.Module):
    """Ultra-compact version for very small datasets."""

    def __init__(
            self,
            in_channels: int = 4,
            base_channels: int = 6,  # Very small base
            dropout_rate: float = 0.7,  # Aggressive dropout
            out_features: int = 6,  # Aggressive dropout
    ):
        super().__init__()
        print("Loading PetNetCyl3D Compact (minimal overfitting)")

        self.backbone = nn.Sequential(
            Conv3dCircW(in_channels, base_channels, 3, (1, 1, 1)),
            nn.GroupNorm(2, base_channels),
            nn.GELU(),
            nn.Dropout3d(0.2),

            Conv3dCircW(base_channels, base_channels * 2, 3, (1, 2, 2)),
            nn.GroupNorm(2, base_channels * 2),
            nn.GELU(),
            nn.Dropout3d(0.3),

            Conv3dCircW(base_channels * 2, base_channels * 4, 3, (1, 2, 2)),
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
        print("Loading PetNetCyl3D (Full Features TorchScript Compatible)")

        # Backbone with stride-based downsampling in H,W
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=(1, 1, 1), padding=1),
            nn.GroupNorm(2, base_channels),
            nn.GELU(),
            nn.Dropout3d(0.2),

            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(2, base_channels * 2),
            nn.GELU(),
            nn.Dropout3d(0.3),

            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(4, base_channels * 4),
            nn.GELU(),
            nn.Dropout3d(0.4),
        )

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
        dummy = torch.zeros(1, C, T, H, W)
        with torch.no_grad():
            out = self.backbone(dummy)
        return out.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.head(x)




# -----------------------------------------------------------
# 
# -----------------------------------------------------------
import torch
import torch.nn as nn

class PETVisionTransformer(nn.Module):
    """Simple Vision Transformer for 5D PET data (B, C, T, H, W)"""
    
    def __init__(
        self,
        input_shape=(4, 3, 207, 41),  # (C, T, H, W)
        patch_size=(1, 8, 8),         # (T, H, W) patch dimensions
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_ratio=4.0,
        dropout=0.1,
        out_features=6,
    ):
        super().__init__()
        
        C, T, H, W = input_shape
        patch_t, patch_h, patch_w = patch_size
        
        # Calculate number of patches
        self.num_patches_t = T // patch_t
        self.num_patches_h = H // patch_h  
        self.num_patches_w = W // patch_w
        self.num_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w
        
        print(f"ViT Config: {self.num_patches} patches ({self.num_patches_t}×{self.num_patches_h}×{self.num_patches_w})")
        
        # Patch embedding: conv3d to extract patches and project to embed_dim
        self.patch_embed = nn.Conv3d(
            C, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norm before final prediction
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head - global average pooling + linear
        self.head = nn.Linear(embed_dim, out_features)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Extract patches and embed: (B, C, T, H, W) -> (B, embed_dim, num_patches_t, num_patches_h, num_patches_w)
        x = self.patch_embed(x)
        
        # Flatten spatial dimensions: (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to sequence format: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)
        
        # Classification head
        return self.head(x)


class TransformerBlock(nn.Module):
    """Standard transformer block with multi-head attention and MLP"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x):
        # Multi-head attention with residual connection
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # MLP with residual connection  
        x = x + self.mlp(self.norm2(x))
        return x


class MLP(nn.Module):
    """Simple MLP with GELU activation"""
    
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# -----------------------------------------------------------
# Test
# -----------------------------------------------------------
if __name__ == "__main__":
    # Test regular model
    B, C, T, H, W = 4, 2, 3, 207, 41

    model = PetNetCyl3D(in_channels=C, base_channels=8, num_layers=3)
    x = torch.randn(B, C, T, H, W)
    y_pred = model(x)
    print(f"Regular model output shape: {y_pred.shape}")

    compact_model = PetNetCyl3DCompact(in_channels=C)
    y_pred_compact = compact_model(x)
    print(f"Compact model output shape: {y_pred_compact.shape}")

    full_model = PetNetCyl3DFullFeatures(in_channels=C, base_channels=6, out_features=6)
    y_pred_full = full_model(x)
    print(f"Full model output shape: {y_pred_full.shape}")

    enhanced_model = PetNetCyl3DFullFeaturesEnhanced(in_channels=C, base_channels=6, out_features=6, input_shape=(C,T,H,W))
    y_pred_enhanced = enhanced_model(x)
    print(f"Enhanced model output shape: {y_pred_enhanced.shape}")

    split_model = SplitPetNetCyl3DFullFeaturesEnhanced(in_channels=C, base_channels=6, out_features=6, input_shape=(C,T,H,W))
    y_pred_split = split_model(x)
    print(f"Split Residual model output shape: {y_pred_split.shape}")

    transformer_model = PETVisionTransformer(input_shape=(C,T,H,W), patch_size=(1,8,8), embed_dim=128, num_heads=4, num_layers=4)
    y_pred_transformer = transformer_model(x)
    print(f"Transformer model output shape: {y_pred_transformer.shape}")


    # Count parameters
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Regular model parameters: {count_params(model):,}")
    print(f"Compact model parameters: {count_params(compact_model):,}")
    print(f"Full model parameters: {count_params(full_model):,}")
    print(f"Enhanced model parameters: {count_params(enhanced_model):,}")
    print(f"Split Residual model parameters: {count_params(split_model):,}")
    print(f"Transformer model parameters: {count_params(transformer_model):,}")

