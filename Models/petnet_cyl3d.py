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
# PetNetCyl3D with custom output pooling dimensions
# -----------------------------------------------------------
class PetNetCyl3DCustomPool(nn.Module):
    """Ultra-compact version with custom output pooling dimensions."""

    def __init__(
            self,
            in_channels: int = 4,
            base_channels: int = 6,  # Very small base
            dropout_rate: float = 0.7,  # Aggressive dropout
            out_features: int = 6,
            output_timesteps: int = 8,  # T dimension for output
            output_height: int = 16,    # H dimension for output  
            output_width: int = 16,     # W dimension for output
    ):
        super().__init__()
        print(f"Loading PetNetCyl3D Custom Pool (output: {output_timesteps}x{output_height}x{output_width})")

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

        # Custom adaptive pooling to specified dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool3d((output_timesteps, output_height, output_width))
        
        # Calculate flattened feature size
        self.flattened_features = base_channels * 4 * output_timesteps * output_height * output_width

        self.head = nn.Sequential(
            nn.Linear(self.flattened_features, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, out_features)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptive_pool(x)  # Shape: (batch, channels, T, H, W)
        x = x.flatten(1)           # Shape: (batch, channels * T * H * W)
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
# Compact, full features, attention last layer
# -----------------------------------------------------------
class PetNetCyl3DAttentionFull(nn.Module):
    """Attention version that keeps ALL attended features (TorchScript-compatible)."""

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 6,
        input_shape=(2, 3, 207, 41),  # (C, T, H, W)
        dropout_rate: float = 0.5,
        out_features: int = 6,
        attn_heads: int = 1,
    ):
        super().__init__()
        print("Loading PetNetCyl3D (Attention Full Feature Map, TorchScript Compatible)")

        # Backbone with stride-based downsampling
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

        # Precompute shape
        self.flatten_dim, self.feature_dim, self.num_tokens = self._compute_flatten_stats(in_channels, input_shape)
        print(f"[Init] Flatten: {self.flatten_dim}, Tokens={self.num_tokens}, Feature_dim={self.feature_dim}")

        # Attention projections
        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=False)

        self.attn_heads = attn_heads
        self.scale = (self.feature_dim // attn_heads) ** -0.5

        # Final head takes ALL tokens (N * C)
        self.head = nn.Sequential(
            nn.Linear(self.num_tokens * self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, out_features)
        )

    def _compute_flatten_stats(self, in_channels, input_shape):
        """Compute feature dims and token count (TorchScript safe)."""
        C, T, H, W = input_shape
        dummy = torch.zeros(1, C, T, H, W)
        with torch.no_grad():
            out = self.backbone(dummy)  # (1, C', T', H', W')
        _, C_out, T_out, H_out, W_out = out.shape
        num_tokens = T_out * H_out * W_out
        flatten_dim = num_tokens * C_out
        return flatten_dim, C_out, num_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = self.backbone(x)  # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        N = T * H * W

        # Flatten tokens
        x = x.view(B, C, N).transpose(1, 2)  # (B, N, C)

        # Attention
        Q = self.q_proj(x)  # (B, N, C)
        K = self.k_proj(x)  # (B, N, C)
        V = self.v_proj(x)  # (B, N, C)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, N, N)
        attn_weights = torch.softmax(attn_scores, dim=-1)            # (B, N, N)
        attended = torch.bmm(attn_weights, V)                       # (B, N, C)

        # Keep ALL features -> flatten (B, N*C)
        flat = attended.reshape(B, N * C)

        # Fully connected head
        return self.head(flat)


# -----------------------------------------------------------
# Depthwise Separable Conv3D implementation
# -----------------------------------------------------------
class DepthwiseSeparableConv3d(nn.Module):
    """
    Depthwise Separable Conv3d: depthwise conv followed by pointwise conv.
    Reduces parameters from (in_ch * out_ch * k^3) to (in_ch * k^3 + in_ch * out_ch).
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(1, 1, 1), padding=1, bias=False):
        super().__init__()
        
        # Depthwise convolution (groups = in_channels means each input channel gets its own filter)
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,  # Key: groups = in_channels
            bias=False
        )
        
        # Pointwise convolution (1x1x1 conv to mix channels)
        self.pointwise = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# -----------------------------------------------------------
# PetNetCyl3D with Depthwise Separable Convolutions
# -----------------------------------------------------------
class PetNetCyl3DDepthwise(nn.Module):
    """
    Compact version using depthwise separable convolutions for parameter efficiency.
    Based on PetNetCyl3DFullFeatures but with depthwise separable convs.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 6,
        input_shape=(2, 3, 207, 41),  # (C, T, H, W) so we can compute flatten_dim
        dropout_rate: float = 0.5,
        out_features: int = 6,
    ):
        super().__init__()
        print("Loading PetNetCyl3D (Depthwise Separable Convolutions)")

        # First layer uses regular conv (can't use depthwise when in_channels < groups)
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.norm1 = nn.GroupNorm(2, base_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout3d(0.2)

        # Subsequent layers use depthwise separable convolutions
        self.conv2 = DepthwiseSeparableConv3d(
            base_channels, base_channels * 2, 
            kernel_size=3, stride=(1, 2, 2), padding=1
        )
        self.norm2 = nn.GroupNorm(2, base_channels * 2)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout3d(0.3)

        self.conv3 = DepthwiseSeparableConv3d(
            base_channels * 2, base_channels * 4, 
            kernel_size=3, stride=(1, 2, 2), padding=1
        )
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
        dummy = torch.zeros(1, C, T, H, W)
        with torch.no_grad():
            # Forward through backbone manually
            x = self.act1(self.norm1(self.conv1(dummy)))
            x = self.drop1(x)
            x = self.act2(self.norm2(self.conv2(x)))
            x = self.drop2(x)
            x = self.act3(self.norm3(self.conv3(x)))
            x = self.drop3(x)
        return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through backbone
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
# Windowed Attention Block (TorchScript Compatible)
# -----------------------------------------------------------
class WindowedAttention3D(nn.Module):
    """
    Windowed attention that partitions the feature map into windows along one spatial dimension.
    TorchScript compatible implementation.
    """
    
    def __init__(self, dim: int, window_size: int, num_heads: int = 2, window_dim: str = 'width'):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_dim = window_dim  # 'width' or 'height'
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W) - feature map from conv layer
        """
        B, C, T, H, W = x.shape
        
        # Reshape to (B, T*H*W, C) for attention
        x = x.view(B, C, T * H * W).transpose(1, 2)  # (B, T*H*W, C)
        
        if self.window_dim == 'width':
            # Partition along width dimension
            x = self._partition_width(x, B, T, H, W)
        else:  # height
            # Partition along height dimension  
            x = self._partition_height(x, B, T, H, W)
            
        # Apply attention to each window
        x = self._windowed_attention(x)
        
        if self.window_dim == 'width':
            # Reverse width partitioning
            x = self._reverse_partition_width(x, B, T, H, W)
        else:  # height
            # Reverse height partitioning
            x = self._reverse_partition_height(x, B, T, H, W)
            
        # Reshape back to (B, C, T, H, W)
        x = x.transpose(1, 2).contiguous().view(B, C, T, H, W)
        return x
    
    def _partition_width(self, x: torch.Tensor, B: int, T: int, H: int, W: int) -> torch.Tensor:
        """Partition feature map into windows along width dimension."""
        # x: (B, T*H*W, C)
        # Reshape to (B, T, H, W, C) then partition width
        x = x.view(B, T, H, W, self.dim)
        
        # Pad width if needed
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w))  # Pad width dimension
            W = W + pad_w
            
        # Partition: (B, T, H, num_windows_w, window_size, C)
        num_windows_w = W // self.window_size
        x = x.view(B, T, H, num_windows_w, self.window_size, self.dim)
        
        # Reshape for attention: (B * T * H * num_windows_w, window_size, C)
        x = x.contiguous().view(B * T * H * num_windows_w, self.window_size, self.dim)
        return x
    
    def _partition_height(self, x: torch.Tensor, B: int, T: int, H: int, W: int) -> torch.Tensor:
        """Partition feature map into windows along height dimension."""
        # x: (B, T*H*W, C)  
        # Reshape to (B, T, H, W, C) then partition height
        x = x.view(B, T, H, W, self.dim)
        
        # Pad height if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        if pad_h > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_h))  # Pad height dimension
            H = H + pad_h
            
        # Partition: (B, T, num_windows_h, window_size, W, C)
        num_windows_h = H // self.window_size
        x = x.view(B, T, num_windows_h, self.window_size, W, self.dim)
        
        # Reshape for attention: (B * T * W * num_windows_h, window_size, C)
        x = x.permute(0, 1, 4, 2, 3, 5).contiguous().view(B * T * W * num_windows_h, self.window_size, self.dim)
        return x
    
    def _reverse_partition_width(self, x: torch.Tensor, B: int, T: int, H: int, W: int) -> torch.Tensor:
        """Reverse width partitioning."""
        # Pad width if needed (same as in partition)
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        W_padded = W + pad_w if pad_w > 0 else W
        num_windows_w = W_padded // self.window_size
        
        # x: (B * T * H * num_windows_w, window_size, C)
        # Reshape back: (B, T, H, num_windows_w, window_size, C)
        x = x.contiguous().view(B, T, H, num_windows_w, self.window_size, self.dim)
        
        # Merge windows: (B, T, H, W_padded, C)
        x = x.contiguous().view(B, T, H, W_padded, self.dim)
        
        # Remove padding if added
        if pad_w > 0:
            x = x[:, :, :, :W, :]
            
        # Flatten spatial dims: (B, T*H*W, C)
        x = x.contiguous().view(B, T * H * W, self.dim)
        return x
    
    def _reverse_partition_height(self, x: torch.Tensor, B: int, T: int, H: int, W: int) -> torch.Tensor:
        """Reverse height partitioning."""
        # Pad height if needed (same as in partition)
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        H_padded = H + pad_h if pad_h > 0 else H
        num_windows_h = H_padded // self.window_size
        
        # x: (B * T * W * num_windows_h, window_size, C)
        # Reshape back: (B, T, W, num_windows_h, window_size, C)
        x = x.contiguous().view(B, T, W, num_windows_h, self.window_size, self.dim)
        
        # Permute and merge: (B, T, H_padded, W, C)
        x = x.permute(0, 1, 3, 4, 2, 5).contiguous().view(B, T, H_padded, W, self.dim)
        
        # Remove padding if added
        if pad_h > 0:
            x = x[:, :, :H, :, :]
            
        # Flatten spatial dims: (B, T*H*W, C)
        x = x.contiguous().view(B, T * H * W, self.dim)
        return x
    
    def _windowed_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention within each window."""
        # x: (num_windows, window_size, C)
        num_windows, window_size, _ = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (num_windows, window_size, 3*C)
        qkv = qkv.view(num_windows, window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, num_windows, num_heads, window_size, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (num_windows, num_heads, window_size, window_size)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        x = torch.matmul(attn, v)  # (num_windows, num_heads, window_size, head_dim)
        x = x.transpose(1, 2).contiguous().view(num_windows, window_size, self.dim)
        
        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------------------------------------
# PetNetCyl3D with Windowed Attention and Multiple FC Heads
# -----------------------------------------------------------
class PetNetCyl3DWindowedAttention(nn.Module):
    """
    PetNet with windowed attention applied after first convolution.
    Two successive attention blocks: width-windowed, then height-windowed.
    Multiple specialized FC heads for each output channel.
    TorchScript compatible.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 6,
        input_shape=(2, 3, 207, 41),  # (C, T, H, W) - original channels before positional
        dropout_rate: float = 0.5,
        out_features: int = 6,
        window_size: int = 8,  # Window size for attention
        attn_heads: int = 2,
        normalize_positions: bool = True,
    ):
        super().__init__()
        print("Loading PetNetCyl3D (Windowed Attention, Multiple FC Heads, TorchScript Compatible)")
        
        self.normalize_positions = normalize_positions
        self.out_features = out_features

        # First convolution (input channels = 4 after adding positional features)
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.norm1 = nn.GroupNorm(2, base_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout3d(0.2)

        # Windowed attention blocks after first conv
        self.width_attention = WindowedAttention3D(
            dim=base_channels, 
            window_size=window_size, 
            num_heads=attn_heads, 
            window_dim='width'
        )
        
        self.height_attention = WindowedAttention3D(
            dim=base_channels, 
            window_size=window_size, 
            num_heads=attn_heads, 
            window_dim='height'
        )

        # Remaining convolution layers
        self.conv2 = nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.norm2 = nn.GroupNorm(2, base_channels * 2)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout3d(0.3)

        self.conv3 = nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.norm3 = nn.GroupNorm(4, base_channels * 4)
        self.act3 = nn.GELU()
        self.drop3 = nn.Dropout3d(0.4)

        # Precompute flatten_dim from input shape (with positional features)
        flatten_dim = self._compute_flatten_dim(in_channels, input_shape)
        print(f"[Init] Flattened feature size = {flatten_dim}")

        # Shared feature extraction layers
        self.shared_fc1 = nn.Linear(flatten_dim, 256)
        self.shared_dropout1 = nn.Dropout(dropout_rate)
        self.shared_fc2 = nn.Linear(256, 128)
        self.shared_dropout2 = nn.Dropout(dropout_rate)

        # Individual FC heads for each output channel
        self.fc_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            ) for _ in range(out_features)
        ])

        # Alternative: Simpler individual heads (uncomment to use)
        # self.fc_heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(128, 64),
        #         nn.GELU(),
        #         nn.Dropout(0.3),
        #         nn.Linear(64, 1)
        #     ) for _ in range(out_features)
        # ])

    def _compute_flatten_dim(self, in_channels, input_shape):
        """Compute feature map flatten size at init time (TorchScript safe)."""
        C, T, H, W = input_shape
        # Create dummy input with original channels
        dummy = torch.zeros(1, C, T, H, W)
        
        with torch.no_grad():
            # Forward through full model pipeline
            x = self.act1(self.norm1(self.conv1(dummy)))
            x = self.drop1(x)
            
            # Apply windowed attention blocks
            x = self.width_attention(x)
            x = self.height_attention(x)
            
            # Remaining convolutions
            x = self.act2(self.norm2(self.conv2(x)))
            x = self.drop2(x)
            x = self.act3(self.norm3(self.conv3(x)))
            x = self.drop3(x)
            
        return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # First convolution
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.drop1(x)
        
        # Apply windowed attention: width-windowed, then height-windowed
        x = self.width_attention(x)
        x = self.height_attention(x)
        
        # Remaining convolutions
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.act3(self.norm3(self.conv3(x)))
        x = self.drop3(x)
        
        # Flatten and pass through shared feature extraction
        x = torch.flatten(x, 1)
        
        # Shared feature extraction
        x = self.shared_fc1(x)
        x = self.act1(x)  # Reuse activation
        x = self.shared_dropout1(x)
        
        x = self.shared_fc2(x)
        x = self.act1(x)  # Reuse activation
        x = self.shared_dropout2(x)
        
        # Apply individual FC heads
        outputs = []
        for head in self.fc_heads:
            head_output = head(x)  # Shape: (batch_size, 1)
            outputs.append(head_output)
        
        # Concatenate all head outputs
        final_output = torch.cat(outputs, dim=1)  # Shape: (batch_size, out_features)
        
        return final_output

    def get_head_outputs(self, x: torch.Tensor):
        """
        Get outputs from each head separately (useful for analysis/debugging).
        
        Returns:
            List of tensors, each of shape (batch_size, 1)
        """
        # Forward through shared layers
        x = add_positional_features(x, normalize=self.normalize_positions)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.drop1(x)
        x = self.width_attention(x)
        x = self.height_attention(x)
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.drop2(x)
        x = self.act3(self.norm3(self.conv3(x)))
        x = self.drop3(x)
        x = torch.flatten(x, 1)
        x = self.shared_fc1(x)
        x = self.act1(x)
        x = self.shared_dropout1(x)
        x = self.shared_fc2(x)
        x = self.act1(x)
        x = self.shared_dropout2(x)
        
        # Get individual head outputs
        head_outputs = []
        for head in self.fc_heads:
            head_output = head(x)
            head_outputs.append(head_output)
        
        return head_outputs

    def get_feature_maps(self, x: torch.Tensor, return_attention_maps: bool = False):
        """
        Extract intermediate feature maps for visualization/analysis.
        
        Args:
            x: Input tensor
            return_attention_maps: Whether to return attention maps from windowed attention
        
        Returns:
            Dictionary containing feature maps at different stages
        """
        feature_maps = {}
        
        x = add_positional_features(x, normalize=self.normalize_positions)
        feature_maps['input_with_pos'] = x.clone()
        
        # After first conv
        x = self.act1(self.norm1(self.conv1(x)))
        feature_maps['after_conv1'] = x.clone()
        x = self.drop1(x)
        
        # After attention blocks
        x = self.width_attention(x)
        feature_maps['after_width_attention'] = x.clone()
        x = self.height_attention(x)
        feature_maps['after_height_attention'] = x.clone()
        
        # After remaining convolutions
        x = self.act2(self.norm2(self.conv2(x)))
        feature_maps['after_conv2'] = x.clone()
        x = self.drop2(x)
        
        x = self.act3(self.norm3(self.conv3(x)))
        feature_maps['after_conv3'] = x.clone()
        x = self.drop3(x)
        
        # Flattened features
        x = torch.flatten(x, 1)
        feature_maps['flattened'] = x.clone()
        
        # After shared FC layers
        x = self.shared_fc1(x)
        x = self.act1(x)
        x = self.shared_dropout1(x)
        feature_maps['after_shared_fc1'] = x.clone()
        
        x = self.shared_fc2(x)
        x = self.act1(x)
        x = self.shared_dropout2(x)
        feature_maps['after_shared_fc2'] = x.clone()
        
        return feature_maps


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

    full_attn_model = PetNetCyl3DAttentionFull(in_channels=C, base_channels=6, out_features=6)
    y_pred_full_attm = full_attn_model(x)
    print(f"Full attention model output shape: {y_pred_full_attm.shape}")

    # Standard depthwise model
    depthwise_model = PetNetCyl3DDepthwise(in_channels=C, base_channels=6, out_features=6)
    y_pred_dw = depthwise_model(x)
    print(f"Depthwise model output shape: {y_pred_dw.shape}")

    windowed_model = PetNetCyl3DWindowedAttention(in_channels=C, base_channels=6, 
        out_features=6, window_size=8, attn_heads=2)
    y_pred_window = depthwise_model(x)
    print(f"Windowed model output shape: {y_pred_window.shape}")

    custom_pool = PetNetCyl3DCustomPool(in_channels=C, base_channels=6, out_features=6,
                                       output_timesteps=T, output_height=H, output_width=W)
    y_pred_custom = custom_pool(x)
    print(f"Custom pool model output shape: {y_pred_custom.shape}")

    # Count parameters
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Regular model parameters: {count_params(model):,}")
    print(f"Compact model parameters: {count_params(compact_model):,}")
    print(f"Full model parameters: {count_params(full_model):,}")
    print(f"Full attention model parameters: {count_params(full_attn_model):,}")
    print(f"DW model parameters: {count_params(depthwise_model):,}")
    print(f"Windowed model parameters: {count_params(windowed_model):,}")
    print(f"Custom pool model parameters: {count_params(custom_pool):,}")

