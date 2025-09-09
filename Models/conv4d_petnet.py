import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CircularPadConv4d(nn.Module):
    """4D convolution with circular padding on width dimension."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_kernel: int = 3,
        bias: bool = False,
    ):
        super().__init__()
        self.spatial_kernel = spatial_kernel
        
        # Use Conv2d for spatial convolution after flattening T*C
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=spatial_kernel,
            stride=1,
            padding=0,
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T*C, H, W)
        B, TC, H, W = x.shape
        
        # Apply circular padding on width (W dimension)
        pad_w = (self.spatial_kernel - 1) // 2
        pad_h = (self.spatial_kernel - 1) // 2
        
        if pad_w > 0:
            # Circular padding on last dimension
            x_padded_w = torch.cat([x[..., -pad_w:], x, x[..., :pad_w]], dim=-1)
        else:
            x_padded_w = x
            
        if pad_h > 0:
            # Zero padding on H dimension
            x_padded = F.pad(x_padded_w, (0, 0, pad_h, pad_h), mode="constant", value=0.0)
        else:
            x_padded = x_padded_w
        
        # Apply 2D convolution
        out = self.conv2d(x_padded)  # (B, out_channels, H_out, W_out)
        
        return out


class ParallelConvBranch(nn.Module):
    """Single convolution branch with specific kernel size."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_kernel: int,
        debug: bool = False
    ):
        super().__init__()
        self.debug = debug
        self.spatial_kernel = spatial_kernel
        
        self.conv = CircularPadConv4d(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_kernel=spatial_kernel
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print(f"Branch (kernel={self.spatial_kernel}) input shape: {x.shape}")
            
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        
        if self.debug:
            print(f"Branch (kernel={self.spatial_kernel}) output shape: {out.shape}")
            
        return out


class ParallelConvBlock(nn.Module):
    """Parallel convolution block with 3 branches (kernel sizes 1, 3, 5)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels_per_branch: int,
        debug: bool = False
    ):
        super().__init__()
        self.debug = debug
        
        # 3 parallel branches with different kernel sizes
        self.branch_1x1 = ParallelConvBranch(
            in_channels, out_channels_per_branch, spatial_kernel=1, debug=debug
        )
        self.branch_3x3 = ParallelConvBranch(
            in_channels, out_channels_per_branch, spatial_kernel=3, debug=debug
        )
        self.branch_5x5 = ParallelConvBranch(
            in_channels, out_channels_per_branch, spatial_kernel=5, debug=debug
        )
        
        # Total output channels
        total_out_channels = 3 * out_channels_per_branch
        
        # Optional 1x1 conv to reduce channels after concatenation
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(total_out_channels, out_channels_per_branch * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels_per_branch * 2),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection if needed
        self.residual_proj = None
        final_out_channels = out_channels_per_branch * 2
        if in_channels != final_out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, final_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(final_out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print(f"ParallelConvBlock input shape: {x.shape}")
        
        residual = x
        
        # Process through parallel branches
        out_1x1 = self.branch_1x1(x)
        out_3x3 = self.branch_3x3(x)
        out_5x5 = self.branch_5x5(x)
        
        # Concatenate along channel dimension
        out = torch.cat([out_1x1, out_3x3, out_5x5], dim=1)
        
        if self.debug:
            print(f"After concatenation: {out.shape}")
        
        # Reduce channels
        out = self.channel_reduce(out)
        
        if self.debug:
            print(f"After channel reduction: {out.shape}")
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        out = out + residual
        
        if self.debug:
            print(f"ParallelConvBlock output shape: {out.shape}")
            
        return out


class Conv4DPetNet(nn.Module):
    """4D Convolutional PET Network with parallel multi-scale branches."""
    
    def __init__(
        self,
        input_channels: int = 6,
        input_time: int = 3,
        base_channels: int = 32,
        output_features: int = 3,
        debug: bool = False
    ):
        super().__init__()
        self.debug = debug
        self.input_channels = input_channels
        self.input_time = input_time
        
        # Calculate input size for first conv (T * C)
        first_conv_in = input_time * input_channels
        
        # Single parallel convolution block that processes the input
        self.parallel_conv = ParallelConvBlock(
            in_channels=first_conv_in,
            out_channels_per_branch=base_channels,
            debug=debug
        )
        
        # Calculate final feature channels after parallel block
        final_channels = base_channels * 2
        
        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Prediction head (3-layer MLP with sparsity)
        self.mlp = nn.Sequential(
            nn.Linear(final_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),  # 30% sparsity
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),  # 30% sparsity
            
            nn.Linear(256, output_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            
        Returns:
            Output tensor of shape (B, output_features)
        """
        if self.debug:
            print(f"Input shape: {x.shape}")
        
        # Convert from (B, C, T, H, W) to (B, T, C, H, W) and make contiguous
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        if self.debug:
            print(f"After permute: {x.shape}")
        
        # Flatten T and C dimensions: (B, T, C, H, W) -> (B, T*C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        
        if self.debug:
            print(f"After flattening T*C: {x.shape}")
        
        # Pass through parallel convolution block
        x = self.parallel_conv(x)
        
        if self.debug:
            print(f"After parallel conv block: {x.shape}")
        
        # Global pooling and flatten
        x = self.global_pool(x)  # (B, features, 1, 1)
        x = x.view(x.size(0), -1)  # (B, features)
        
        if self.debug:
            print(f"After global pool and flatten: {x.shape}")
        
        # Pass through MLP
        x = self.mlp(x)
        
        if self.debug:
            print(f"Final output: {x.shape}")
        
        return x

# Test with dummy data
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Input data shape from original code
    # B, C, T, H, W = 2, 6, 3, 62, 519
    B, C, T, H, W = 8, 2, 3, 207, 41
    x = torch.rand(B, C, T, H, W, device=device)
    
    print("=== Conv4D PET Network Test ===")
    print(f"Input shape: {x.shape}")
    
    # Create model with debug enabled
    model = Conv4DPetNet(
        input_channels=C,
        input_time=T,
        base_channels=32,
        output_features=3,
        debug=True
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"\nFinal output shape: {output.shape}")
    print(f"Sample output: {output[0]}")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")