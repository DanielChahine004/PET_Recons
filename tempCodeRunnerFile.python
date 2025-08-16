import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple

###############################################################################
# TorchScript-Compatible Global Attention Module with Skip Connection
###############################################################################
class GlobalAttention3D(nn.Module):
    """
    Fully TorchScript-compatible global attention module for 3D feature maps with skip connection.
    """
    
    def __init__(self, in_channels: int = 64, embed_dim: int = 128, output_dim: int = 64, 
                 num_heads: int = 2, max_seq_len: int = 1000):
        super(GlobalAttention3D, self).__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Map from input channels to embedding dimension
        self.channel_proj = nn.Linear(in_channels, embed_dim)
        
        # Manual attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Map from embedding dimension to output channels
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
        # Pre-allocated positional encoding (TorchScript compatible)
        self.register_buffer('pos_encoding', torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.scale = self.head_dim ** -0.5
        
        # Always create skip projection modules (TorchScript compatible)
        self.skip_proj = nn.Conv3d(in_channels, output_dim, kernel_size=1, bias=False)
        self.skip_bn = nn.BatchNorm3d(output_dim)
        self.use_skip_proj = (in_channels != output_dim)
        
        # Learnable gate for skip connection strength
        self.gate = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, channels, D, H, W)
        Returns: (batch_size, output_dim, D, H, W)
        """
        batch_size, channels, D, H, W = x.shape
        seq_len = D * H * W
        
        # Store input for skip connection
        skip_input = x
        
        # Reshape to treat spatial positions as sequence tokens
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous()
        x_flat = x_flat.view(batch_size, seq_len, channels)
        
        # Project to embedding dimension
        x_flat = self.channel_proj(x_flat)
        
        # Add positional encoding (slice to current sequence length)
        x_flat = x_flat + self.pos_encoding[:, :seq_len, :]
        
        # Manual multi-head attention
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Project to output channels per token
        output = self.output_proj(attn_output)
        
        # Reshape back to conv format
        output = output.permute(0, 2, 1)
        output = output.view(batch_size, self.output_dim, D, H, W)
        
        # Skip connection (always apply projection, but conditionally use it)
        projected_skip = self.skip_bn(self.skip_proj(skip_input))
        if self.use_skip_proj:
            skip_input = projected_skip
        
        # Add skip connection with learnable gate
        output = output + self.gate * skip_input
        
        return output


###############################################################################
# TorchScript-Compatible Normalized Tensor Train Module
###############################################################################
class NormalizedDirectTensorTrain3D(nn.Module):
    """
    TorchScript-compatible direct tensor train with pre-normalization for stability.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int, int], rank: int = 64, norm_type: int = 0):
        super(NormalizedDirectTensorTrain3D, self).__init__()
        
        self.channels, self.D, self.H, self.W = input_shape
        self.rank = rank
        self.norm_type = norm_type
        
        # Always create all normalization types (TorchScript compatible)
        self.batch_norm = nn.BatchNorm3d(self.channels)
        self.layer_norm = nn.LayerNorm([self.D, self.H, self.W])
        num_groups = min(8, self.channels)
        self.group_norm = nn.GroupNorm(num_groups, self.channels)
        self.instance_norm = nn.InstanceNorm3d(self.channels)
        self.identity = nn.Identity()
        
        # TT cores with improved initialization
        self.core1 = nn.Parameter(torch.randn(1, self.D, rank) * (2.0 / (self.D + rank)) ** 0.5)
        self.core2 = nn.Parameter(torch.randn(rank, self.H, rank) * (2.0 / (self.H + 2*rank)) ** 0.5)
        self.core3 = nn.Parameter(torch.randn(rank, self.W, 1) * (2.0 / (self.W + rank)) ** 0.5)
        
        # Activation function for output stabilization
        self.activation = nn.GELU()
        
        self.output_size = rank * 3
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization then tensor train decomposition.
        """
        batch_size, channels, D, H, W = x.shape
        
        # Apply normalization based on norm_type (TorchScript compatible)
        if self.norm_type == 0:  # 'batch'
            x_norm = self.batch_norm(x)
        elif self.norm_type == 1:  # 'layer'
            x_norm = self.layer_norm(x)
        elif self.norm_type == 2:  # 'group'
            x_norm = self.group_norm(x)
        elif self.norm_type == 3:  # 'instance'
            x_norm = self.instance_norm(x)
        else:  # 'none'
            x_norm = self.identity(x)
        
        # Reshape for processing
        x_flat = x_norm.reshape(batch_size * channels, D, H, W)
        
        # Mode-wise contractions with improved numerical stability
        # Mode-1: contract over D dimension
        mode1_features = torch.einsum('bdhw,idr->bhwr', x_flat, self.core1)
        mode1_summary = torch.mean(mode1_features, dim=(1, 2))  # (B*C, rank)
        
        # Mode-2: contract over H dimension  
        mode2_features = torch.einsum('bdhw,rhs->bdwrs', x_flat, self.core2)
        mode2_summary = torch.mean(mode2_features, dim=(1, 2, 4))  # (B*C, rank)
        
        # Mode-3: contract over W dimension
        mode3_features = torch.einsum('bdhw,rwi->bdhr', x_flat, self.core3)
        mode3_summary = torch.mean(mode3_features, dim=(1, 2))  # (B*C, rank)
        
        # Concatenate mode summaries
        core_summaries = torch.cat([mode1_summary, mode2_summary, mode3_summary], dim=1)
        
        # Apply activation function for stabilization and non-linearity
        core_summaries = self.activation(core_summaries)
        
        # Reshape back to batch format
        output = core_summaries.reshape(batch_size, channels * self.output_size)
        
        return output


###############################################################################
# TorchScript-Compatible 3D Residual Block
###############################################################################
class ResidualBlock3D(nn.Module):
    """
    TorchScript-compatible 3D adaptation of a 2-layer residual block.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int, int] = (1, 2, 2)):
        super(ResidualBlock3D, self).__init__()

        # First 3D conv
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        # Second 3D conv
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Post skip-connection processing layer
        self.conv_post = nn.Conv3d(out_channels, out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=0,
                                   bias=False)
        self.bn_post = nn.BatchNorm3d(out_channels)

        self.activation = nn.GELU()

        # Always create shortcut modules (TorchScript compatible)
        self.shortcut_conv = nn.Conv3d(in_channels, out_channels,
                                      kernel_size=1,
                                      stride=stride,
                                      bias=False)
        self.shortcut_bn = nn.BatchNorm3d(out_channels)
        self.use_shortcut = (stride != (1, 1, 1) or in_channels != out_channels)

    def _apply_circular_padding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply circular padding for width axis and regular padding for time and height.
        """
        # Circular padding for width, constant for others
        x = torch.nn.functional.pad(x, (1, 1, 0, 0, 0, 0), mode='circular')
        x = torch.nn.functional.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0.0)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        out = self._apply_circular_padding(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)

        out = self._apply_circular_padding(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut (always compute, conditionally use)
        shortcut_out = self.shortcut_bn(self.shortcut_conv(x))
        if self.use_shortcut:
            x = shortcut_out

        # Residual connection
        out += x

        # Post skip-connection processing
        out = self._apply_circular_padding(out)
        out = self.conv_post(out)
        out = self.bn_post(out)
        out = self.activation(out)
        
        return out


###############################################################################
# TorchScript-Compatible PetNetImproved3D
###############################################################################
class PetNetImproved3D(nn.Module):
    """
    Fully TorchScript-compatible PetNet with static tensor train initialization.
    """

    def __init__(self, num_classes: int = 6, tt_rank: int = 32, norm_type: int = 0, 
                 input_shape: Tuple[int, int, int, int] = (2, 3, 207, 41)):
        super(PetNetImproved3D, self).__init__()

        self.tt_rank = tt_rank
        self.norm_type = norm_type
        self.input_shape = input_shape

        # Initial 3D conv: 2 => 16 channels
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=16,
                                 kernel_size=3, stride=(1, 1, 1),
                                 padding=1, bias=False)
        self.bn_in = nn.BatchNorm3d(16)
        self.activation = nn.GELU()

        # Residual blocks
        self.layer1 = ResidualBlock3D(16, 32, stride=(1, 2, 2))
        self.layer2 = ResidualBlock3D(32, 64, stride=(1, 2, 2))

        # Global attention after layer2
        # Calculate max_seq_len based on worst-case scenario after layer2
        max_seq_len = self._estimate_max_seq_len(input_shape)
        self.global_attention = GlobalAttention3D(
            in_channels=64, 
            embed_dim=128, 
            output_dim=64,
            num_heads=8,
            max_seq_len=max_seq_len
        )

        # Remaining residual blocks
        self.layer3 = ResidualBlock3D(64, 128, stride=(1, 2, 2))
        self.layer4 = ResidualBlock3D(128, 256, stride=(1, 2, 2))

        # **STATIC TENSOR TRAIN INITIALIZATION**
        feature_shape = self._compute_feature_shape(input_shape)
        self.tensor_train = NormalizedDirectTensorTrain3D(
            feature_shape, 
            rank=self.tt_rank,
            norm_type=self.norm_type
        )
        
        self.dropout = nn.Dropout(0.3)

        # Compute FC input features
        fc_in_features = feature_shape[0] * self.tensor_train.output_size
        self.fc1 = nn.Linear(fc_in_features, 1024, bias=True)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

        self._initialize_weights()

    def _estimate_max_seq_len(self, input_shape: Tuple[int, int, int, int]) -> int:
        """
        Estimate maximum sequence length for positional encoding buffer.
        """
        C, T, H, W = input_shape
        # Conservative estimate: assume minimal downsampling
        # After 2 layers with stride (1,2,2), roughly: T, H//4, W//4
        max_t = T + 10  # Add buffer
        max_h = (H // 4) + 10
        max_w = (W // 4) + 10
        return max_t * max_h * max_w

    def _compute_feature_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Compute feature map shape by running a dummy tensor through the conv layers.
        TorchScript compatible version.
        """
        C, T, H, W = input_shape
        
        with torch.no_grad():
            # Create dummy input
            dummy = torch.zeros(1, C, T, H, W)
            
            # Run through conv layers in order
            x = self.conv_in(dummy)
            x = self.bn_in(x)
            x = self.activation(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.global_attention(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            # Return shape without batch dimension
            return (int(x.shape[1]), int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        x: expected shape (batch_size, 2, T, H, W)
        """
        # Initial conv
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.activation(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)

        # Global attention with skip connection
        x = self.global_attention(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Ensure tensor is contiguous before tensor train
        x = x.contiguous()

        # Apply normalized tensor train decomposition
        x = self.tensor_train(x)

        # FC layers
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def _initialize_weights(self) -> None:
        """
        Kaiming (He) Initialization for Conv3d and Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


###############################################################################
# Training Script
###############################################################################
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B = 8
    C = 2
    T = 3 
    H = 207
    W = 41
    CLASSES = 6

    # Model instantiation with TorchScript-compatible static initialization
    # norm_type: 0=batch, 1=layer, 2=group, 3=instance, 4=none
    model = PetNetImproved3D(
        num_classes=CLASSES, 
        tt_rank=32, 
        norm_type=0,  # batch normalization
        input_shape=(C, T, H, W)
    ).to(device)
    
    # Test TorchScript compatibility
    try:
        scripted_model = torch.jit.script(model)
        print("✅ Model is TorchScript compatible!")
    except Exception as e:
        print(f"❌ TorchScript error: {e}")
    
    # Test a dummy pass
    dummy_input = torch.randn(B, C, T, H, W).to(device)
    dummy_target = torch.randn(B, CLASSES).to(device)
    
    # Test both regular and scripted model
    output_regular = model(dummy_input)
    if 'scripted_model' in locals():
        output_scripted = scripted_model(dummy_input)
        print(f"Regular model output shape: {output_regular.shape}")
        print(f"Scripted model output shape: {output_scripted.shape}")
        print(f"Outputs match: {torch.allclose(output_regular, output_scripted, atol=1e-5)}")
    
    # Print parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,}")

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    total_time = 0.0
    epochs = 500

    for epoch in range(epochs):
        optimizer.zero_grad()
        start_time = time.time()
        output = model(dummy_input)
        end_time = time.time()
        forward_time = end_time - start_time
        total_time += forward_time

        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Forward Time: {forward_time:.6f}s")

    average_time = total_time / epochs
    print(f"Average Forward Pass Time: {average_time:.6f}s")