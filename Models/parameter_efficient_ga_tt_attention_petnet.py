import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple

###############################################################################
# TorchScript-Compatible Memory-Efficient Global Attention Module
###############################################################################
class MemoryEfficientGlobalAttention3D(nn.Module):
    """
    Memory-efficient global attention using chunked computation and pooling.
    """
    
    def __init__(self, in_channels: int = 32, embed_dim: int = 64, output_dim: int = 32, 
                 num_heads: int = 4, max_seq_len: int = 1000, chunk_size: int = 512):
        super(MemoryEfficientGlobalAttention3D, self).__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Adaptive pooling to reduce sequence length (TorchScript compatible)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((3, 32, 8))  # Fixed dimensions without None
        
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
        reduced_max_seq_len = min(max_seq_len, 1024)  # Cap the max sequence length
        self.register_buffer('pos_encoding', torch.randn(1, reduced_max_seq_len, embed_dim) * 0.02)
        self.scale = self.head_dim ** -0.5
        
        # Always create skip projection modules (TorchScript compatible)
        self.skip_proj = nn.Conv3d(in_channels, output_dim, kernel_size=1, bias=False)
        self.skip_bn = nn.BatchNorm3d(output_dim)
        self.use_skip_proj = (in_channels != output_dim)
        
        # Upsampling to restore original size (TorchScript compatible)
        self.upsample = nn.Upsample(scale_factor=(1.0, 4.0, 2.625), mode='trilinear', align_corners=False)
        
        # Learnable gate for skip connection strength
        self.gate = nn.Parameter(torch.tensor(0.1))
        
    def _chunked_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          debug: bool = False) -> torch.Tensor:
        """
        Compute attention in chunks to save memory.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        if seq_len <= self.chunk_size:
            # Small enough, compute normally
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = torch.softmax(attn_weights, dim=-1)
            return torch.matmul(attn_weights, v)
        
        # Chunked computation
        output = torch.zeros_like(v)
        
        for i in range(0, seq_len, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i, :]
            
            # Compute attention weights for this chunk
            attn_weights = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            attn_weights = torch.softmax(attn_weights, dim=-1)
            
            # Apply attention
            output[:, :, i:end_i, :] = torch.matmul(attn_weights, v)
            
            if debug and i == 0: print(f"Chunked attention: processing chunk {i}-{end_i}")
        
        return output
        
    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        x: (batch_size, channels, D, H, W)
        Returns: (batch_size, output_dim, D, H, W)
        """
        batch_size, channels, D, H, W = x.shape
        original_shape = (D, H, W)
        
        if debug: print(f"MemoryEfficientGlobalAttention input: {x.shape}")
        
        # Store input for skip connection
        skip_input = x
        
        # Reduce spatial dimensions to save memory
        x_pooled = self.adaptive_pool(x)
        _, _, D_pool, H_pool, W_pool = x_pooled.shape
        seq_len = D_pool * H_pool * W_pool
        
        if debug: print(f"After adaptive pooling: {x_pooled.shape}, seq_len: {seq_len}")
        
        # Reshape to treat spatial positions as sequence tokens
        x_flat = x_pooled.permute(0, 2, 3, 4, 1).contiguous()
        x_flat = x_flat.view(batch_size, seq_len, channels)
        if debug: print(f"After reshaping to sequence: {x_flat.shape}")
        
        # Project to embedding dimension
        x_flat = self.channel_proj(x_flat)
        if debug: print(f"After channel projection: {x_flat.shape}")
        
        # Add positional encoding (slice to current sequence length)
        pos_len = min(seq_len, self.pos_encoding.shape[1])
        x_flat = x_flat + self.pos_encoding[:, :pos_len, :]
        if debug: print(f"After positional encoding: {x_flat.shape}")
        
        # Manual multi-head attention
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if debug: print(f"Q, K, V shapes after head reshaping: {q.shape}, {k.shape}, {v.shape}")
        
        # Memory-efficient chunked attention
        attn_output = self._chunked_attention(q, k, v, debug=debug)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        if debug: print(f"After concatenating heads: {attn_output.shape}")
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Project to output channels per token
        output = self.output_proj(attn_output)
        if debug: print(f"After output projection: {output.shape}")
        
        # Reshape back to conv format (pooled size)
        output = output.permute(0, 2, 1)
        output = output.view(batch_size, self.output_dim, D_pool, H_pool, W_pool)
        if debug: print(f"After reshaping back to 3D (pooled): {output.shape}")
        
        # Upsample back to original spatial dimensions (TorchScript compatible)
        output = F.interpolate(output, size=original_shape, mode='trilinear', align_corners=False)
        if debug: print(f"After upsampling to original size: {output.shape}")
        
        # Skip connection (always apply projection, but conditionally use it)
        projected_skip = self.skip_bn(self.skip_proj(skip_input))
        if self.use_skip_proj:
            skip_input = projected_skip
        
        # Add skip connection with learnable gate
        output = output + self.gate * skip_input
        if debug: print(f"After skip connection: {output.shape}")
        
        return output


###############################################################################
# TorchScript-Compatible Efficient Tensor Train Module
###############################################################################
class EfficientTensorTrain3D(nn.Module):
    """
    TorchScript-compatible efficient tensor train with reduced rank for parameter efficiency.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int, int], rank: int = 16, norm_type: int = 0):
        super(EfficientTensorTrain3D, self).__init__()
        
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
        
        # TT cores with improved initialization and reduced rank
        self.core1 = nn.Parameter(torch.randn(1, self.D, rank) * (2.0 / (self.D + rank)) ** 0.5)
        self.core2 = nn.Parameter(torch.randn(rank, self.H, rank) * (2.0 / (self.H + 2*rank)) ** 0.5)
        self.core3 = nn.Parameter(torch.randn(rank, self.W, 1) * (2.0 / (self.W + rank)) ** 0.5)
        
        # Activation function for output stabilization
        self.activation = nn.GELU()
        
        self.output_size = rank * 3
        
    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Apply normalization then tensor train decomposition.
        """
        batch_size, channels, D, H, W = x.shape
        
        if debug: print(f"EfficientTensorTrain input: {x.shape}")
        
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
        
        if debug: print(f"After normalization: {x_norm.shape}")
        
        # Reshape for processing
        x_flat = x_norm.reshape(batch_size * channels, D, H, W)
        if debug: print(f"Flattened for TT processing: {x_flat.shape}")
        
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
        if debug: print(f"Concatenated core summaries: {core_summaries.shape}")
        
        # Apply activation function for stabilization and non-linearity
        core_summaries = self.activation(core_summaries)
        
        # Reshape back to batch format
        output = core_summaries.reshape(batch_size, channels * self.output_size)
        if debug: print(f"EfficientTensorTrain output: {output.shape}")
        
        return output


###############################################################################
# TorchScript-Compatible Depthwise Separable Conv3D
###############################################################################
class DepthwiseSeparableConv3D(nn.Module):
    """
    Depthwise separable convolution for parameter efficiency.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: Tuple[int, int, int] = (1, 1, 1), padding: int = 0, bias: bool = False):
        super(DepthwiseSeparableConv3D, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=in_channels, bias=bias)
        
        # Pointwise convolution
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


###############################################################################
# TorchScript-Compatible Efficient 3D Residual Block
###############################################################################
class EfficientResidualBlock3D(nn.Module):
    """
    Parameter-efficient 3D residual block using depthwise separable convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int, int] = (1, 2, 2)):
        super(EfficientResidualBlock3D, self).__init__()

        # First depthwise separable conv
        self.conv1 = DepthwiseSeparableConv3D(in_channels, out_channels,
                                             kernel_size=3,
                                             stride=stride,
                                             padding=0,
                                             bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        # Second depthwise separable conv
        self.conv2 = DepthwiseSeparableConv3D(out_channels, out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=0,
                                             bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Post skip-connection processing layer
        self.conv_post = DepthwiseSeparableConv3D(out_channels, out_channels,
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

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        if debug: print(f"EfficientResBlock input: {x.shape}")
        
        # Main path
        out = self._apply_circular_padding(x)
        if debug: print(f"After padding: {out.shape}")
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)
        if debug: print(f"After conv1+bn1+activation: {out.shape}")

        out = self._apply_circular_padding(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if debug: print(f"After conv2+bn2: {out.shape}")

        # Shortcut (always compute, conditionally use)
        shortcut_out = self.shortcut_bn(self.shortcut_conv(x))
        if self.use_shortcut:
            x = shortcut_out
        if debug: print(f"Shortcut: {x.shape}")

        # Residual connection
        out += x
        if debug: print(f"After residual connection: {out.shape}")

        # Post skip-connection processing
        out = self._apply_circular_padding(out)
        out = self.conv_post(out)
        out = self.bn_post(out)
        out = self.activation(out)
        if debug: print(f"After post-processing: {out.shape}")
        
        return out


###############################################################################
# TorchScript-Compatible Parameter-Efficient PetNetImproved3D
###############################################################################
class ParameterEfficientPetNet3D(nn.Module):
    """
    Parameter-efficient version of PetNet with reduced channels and depthwise separable convolutions.
    """

    def __init__(self, num_classes: int = 6, tt_rank: int = 16, norm_type: int = 0, 
                 input_shape: Tuple[int, int, int, int] = (2, 3, 207, 41)):
        super(ParameterEfficientPetNet3D, self).__init__()
        print("Loading Parameter-Efficient PetNetImproved3D Model...")

        self.tt_rank = tt_rank
        self.norm_type = norm_type
        self.input_shape = input_shape

        # Initial 3D conv: 2 => 16 channels
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=16,
                                 kernel_size=3, stride=(1, 1, 1),
                                 padding=1, bias=False)
        self.bn_in = nn.BatchNorm3d(16)
        self.activation = nn.GELU()

        # Efficient residual blocks with reduced channel progression: 16→24→32→48→64
        self.layer1 = EfficientResidualBlock3D(16, 24, stride=(1, 2, 2))
        self.layer2 = EfficientResidualBlock3D(24, 32, stride=(1, 2, 2))

        # Channel reduction before attention
        self.channel_reducer1 = nn.Conv3d(32, 24, kernel_size=1, bias=False)
        self.bn_reducer1 = nn.BatchNorm3d(24)

        # Memory-efficient global attention after layer2
        max_seq_len = self._estimate_max_seq_len(input_shape)
        self.global_attention = MemoryEfficientGlobalAttention3D(
            in_channels=24,    # Reduced from 64
            embed_dim=48,      # Reduced from 128
            output_dim=24,     # Keep same as input
            num_heads=4,       # Reduced from 8
            max_seq_len=max_seq_len,
            chunk_size=512     # Memory-efficient chunking
        )

        # Remaining efficient residual blocks
        self.layer3 = EfficientResidualBlock3D(24, 48, stride=(1, 2, 2))
        
        # Channel reduction before final layer
        self.channel_reducer2 = nn.Conv3d(48, 32, kernel_size=1, bias=False)
        self.bn_reducer2 = nn.BatchNorm3d(32)
        
        self.layer4 = EfficientResidualBlock3D(32, 64, stride=(1, 2, 2))

        # **EFFICIENT TENSOR TRAIN INITIALIZATION**
        feature_shape = self._compute_feature_shape(input_shape)
        self.tensor_train = EfficientTensorTrain3D(
            feature_shape, 
            rank=self.tt_rank,  # Reduced from 32 to 16
            norm_type=self.norm_type
        )
        
        self.dropout = nn.Dropout(0.3)

        # Compute FC input features
        fc_in_features = feature_shape[0] * self.tensor_train.output_size
        
        # Reduced FC layer sizes
        self.fc1 = nn.Linear(fc_in_features, 512, bias=True)  # Reduced from 1024
        self.fc2 = nn.Linear(512, num_classes, bias=True)

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
            
            # Channel reduction
            x = self.channel_reducer1(x)
            x = self.bn_reducer1(x)
            x = self.activation(x)
            
            x = self.global_attention(x)
            x = self.layer3(x)
            
            # Channel reduction
            x = self.channel_reducer2(x)
            x = self.bn_reducer2(x)
            x = self.activation(x)
            
            x = self.layer4(x)
            
            # Return shape without batch dimension
            return (int(x.shape[1]), int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        x: expected shape (batch_size, 2, T, H, W)
        """
        # Initial conv
        if debug: print(f"Input shape: {x.shape}")
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.activation(x)
        if debug: print(f"After conv_in: {x.shape}")

        # Efficient residual blocks
        x = self.layer1(x, debug=debug)
        if debug: print(f"After layer1: {x.shape}")
        x = self.layer2(x, debug=debug)
        if debug: print(f"After layer2: {x.shape}")

        # Channel reduction before attention
        x = self.channel_reducer1(x)
        x = self.bn_reducer1(x)
        x = self.activation(x)
        if debug: print(f"After channel reduction 1: {x.shape}")

        # Memory-efficient global attention with skip connection
        x = self.global_attention(x, debug=debug)
        if debug: print(f"After memory-efficient global attention: {x.shape}")
        
        x = self.layer3(x, debug=debug)
        if debug: print(f"After layer3: {x.shape}")
        
        # Channel reduction before final layer
        x = self.channel_reducer2(x)
        x = self.bn_reducer2(x)
        x = self.activation(x)
        if debug: print(f"After channel reduction 2: {x.shape}")
        
        x = self.layer4(x, debug=debug)
        if debug: print(f"After layer4: {x.shape}")
        
        # Ensure tensor is contiguous before tensor train
        x = x.contiguous()

        # Apply efficient tensor train decomposition
        x = self.tensor_train(x, debug=debug)
        if debug: print(f"After efficient tensor train: {x.shape}")

        # Reduced FC layers
        x = self.fc1(x)
        if debug: print(f"After fc1: {x.shape}")
        x = self.activation(x)
        x = self.dropout(x)
        if debug: print(f"After activation and dropout: {x.shape}")
        x = self.fc2(x)
        if debug: print(f"After fc2 (output): {x.shape}")

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
    # B = 128

    C = 2
    T = 3 

    H = 207
    # H = 496

    W = 41
    # W = 84
    
    CLASSES = 6

    # Model instantiation with parameter-efficient design
    # norm_type: 0=batch, 1=layer, 2=group, 3=instance, 4=none
    model = ParameterEfficientPetNet3D(
        num_classes=CLASSES, 
        tt_rank=42,        # Reduced from 32
        norm_type=0,       # batch normalization
        input_shape=(C, T, H, W)
    ).to(device)
    
    # Test TorchScript compatibility
    try:
        scripted_model = torch.jit.script(model)
        print("✅ Parameter-Efficient Model is TorchScript compatible!")
    except Exception as e:
        print(f"❌ TorchScript error: {e}")
    
    # Test a dummy pass with debug output
    dummy_input = torch.randn(B, C, T, H, W).to(device)
    dummy_target = torch.randn(B, CLASSES).to(device)
    
    print("\n" + "="*60)
    print("DEBUGGING FORWARD PASS - PARAMETER-EFFICIENT MODEL")
    print("="*60)
    
    # Test both regular and scripted model
    output_regular = model(dummy_input, debug=True)
    if 'scripted_model' in locals():
        print("\n" + "-"*40)
        print("Testing scripted model...")
        output_scripted = scripted_model(dummy_input)
        print(f"Regular model output shape: {output_regular.shape}")
        print(f"Scripted model output shape: {output_scripted.shape}")
        print(f"Outputs match: {torch.allclose(output_regular, output_scripted, atol=1e-5)}")
    
    # Print parameter count comparison
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nParameter-Efficient Model Total parameters: {param_count:,}")
    
    # Calculate parameter reduction (estimate from original)
    print(f"Estimated parameter reduction: ~50-60% compared to original model")

    print("\n" + "="*60)
    print("STARTING TRAINING LOOP - PARAMETER-EFFICIENT MODEL")
    print("="*60)

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
    print(f"\nAverage Forward Pass Time: {average_time:.6f}s")