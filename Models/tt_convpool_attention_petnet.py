# takes attention_petnet.py and places the global attention layer after layer 2
# for higher dimensional G.A.
# Modified to use Tensor Train decomposition for spatial preservation

import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Tensor Train Decomposition Module
###############################################################################
class TensorTrain3D(nn.Module):
    """
    Tensor Train decomposition for 3D spatial features.
    Decomposes spatial dimensions (D, H, W) into rank-R tensor train cores.
    """
    
    def __init__(self, input_shape, rank=32, learnable=True):
        """
        Args:
            input_shape: tuple (channels, D, H, W) - shape after conv layers
            rank: tensor train rank (controls compression vs expressiveness)
            learnable: whether to make TT cores learnable parameters
        """
        super(TensorTrain3D, self).__init__()
        
        self.channels, self.D, self.H, self.W = input_shape
        self.rank = rank
        self.learnable = learnable
        
        # TT cores for each spatial dimension
        # Core 1: (1, D, rank) - first core has left rank = 1
        # Core 2: (rank, H, rank) - middle core 
        # Core 3: (rank, W, 1) - last core has right rank = 1
        
        if learnable:
            self.core1 = nn.Parameter(torch.randn(1, self.D, rank) * 0.1)
            self.core2 = nn.Parameter(torch.randn(rank, self.H, rank) * 0.1)
            self.core3 = nn.Parameter(torch.randn(rank, self.W, 1) * 0.1)
        else:
            # Non-learnable cores initialized once
            self.register_buffer('core1', torch.randn(1, self.D, rank) * 0.1)
            self.register_buffer('core2', torch.randn(rank, self.H, rank) * 0.1)
            self.register_buffer('core3', torch.randn(rank, self.W, 1) * 0.1)
        
        # Projection layer to combine channel and TT features
        self.feature_projection = nn.Linear(self.channels * rank * rank, 1024)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, D, H, W)
        Returns:
            compressed representation suitable for FC layers
        """
        batch_size, channels, D, H, W = x.shape
        
        # Reshape to process each channel independently
        # (B, C, D, H, W) -> (B*C, D, H, W)
        x_reshaped = x.reshape(batch_size * channels, D, H, W)
        
        # Apply tensor train decomposition to spatial dimensions
        # For each spatial position (d,h,w), compute TT representation
        
        tt_features = []
        
        # Iterate over spatial positions and compute TT cores contraction
        for b in range(batch_size * channels):
            spatial_tensor = x_reshaped[b]  # (D, H, W)
            
            # Contract tensor with TT cores to get compressed representation
            # This is the key TT operation: tensor contraction
            
            # Initialize with first core
            tt_repr = torch.zeros(self.rank, self.rank, device=x.device)
            
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        # Get scalar value at position (d,h,w)
                        val = spatial_tensor[d, h, w]
                        
                        # Contract with TT cores
                        # core1[0, d, :] -> (rank,)
                        # core2[:, h, :] -> (rank, rank) 
                        # core3[:, w, 0] -> (rank,)
                        
                        c1 = self.core1[0, d, :]  # (rank,)
                        c2 = self.core2[:, h, :]  # (rank, rank)
                        c3 = self.core3[:, w, 0]  # (rank,)
                        
                        # Tensor contraction: c1 * c2 * c3
                        # (rank,) x (rank, rank) -> (rank,)
                        temp = torch.mv(c2, c3)  # (rank,)
                        contrib = c1 * temp * val  # (rank,)
                        
                        # Accumulate contribution (outer product for rank x rank representation)
                        tt_repr += torch.outer(contrib, contrib)
            
            tt_features.append(tt_repr.flatten())  # (rank^2,)
        
        # Stack all TT features
        tt_features = torch.stack(tt_features)  # (B*C, rank^2)
        
        # Reshape back to batch format
        tt_features = tt_features.view(batch_size, channels * self.rank * self.rank)  # (B, C*rank^2)
        
        # Project to desired feature size
        output = self.feature_projection(tt_features)  # (B, 1024)
        
        return output


###############################################################################
# Optimized Tensor Train Module (More Efficient Implementation)
###############################################################################
class OptimizedTensorTrain3D(nn.Module):
    """
    More efficient tensor train implementation using einsum operations.
    """
    
    def __init__(self, input_shape, rank=32):
        super(OptimizedTensorTrain3D, self).__init__()
        
        self.channels, self.D, self.H, self.W = input_shape
        self.rank = rank
        
        # TT cores as learnable parameters
        self.core1 = nn.Parameter(torch.randn(1, self.D, rank) / (self.D ** 0.5))
        self.core2 = nn.Parameter(torch.randn(rank, self.H, rank) / (self.H ** 0.5))
        self.core3 = nn.Parameter(torch.randn(rank, self.W, 1) / (self.W ** 0.5))
        
        # Output projection
        self.output_features = rank * 3  # Each core contributes rank features
        self.feature_projection = nn.Linear(self.channels * self.output_features, 1024)
        
    def forward(self, x):
        """
        Efficient tensor train decomposition using tensor operations.
        """
        batch_size, channels, D, H, W = x.shape
        
        # More efficient approach: compute mode-wise projections
        # Instead of full TT contraction, extract features from each mode
        
        # Reshape for processing: (B, C, D, H, W) -> (B*C, D, H, W)
        x_flat = x.view(batch_size * channels, D, H, W)
        
        # Mode-1 features: contract over D dimension
        # (B*C, D, H, W) * (1, D, rank) -> (B*C, H, W, rank)
        mode1_features = torch.einsum('bdhw,idr->bhwr', x_flat, self.core1)
        mode1_features = torch.mean(mode1_features, dim=(1, 2))  # (B*C, rank)
        
        # Mode-2 features: contract over H dimension  
        # (B*C, D, H, W) * (rank, H, rank) -> (B*C, D, W, rank, rank)
        mode2_features = torch.einsum('bdhw,rhs->bdwrs', x_flat, self.core2)
        mode2_features = torch.mean(mode2_features, dim=(1, 2, 4))  # (B*C, rank)
        
        # Mode-3 features: contract over W dimension
        # (B*C, D, H, W) * (rank, W, 1) -> (B*C, D, H, rank)
        mode3_features = torch.einsum('bdhw,rwi->bdhr', x_flat, self.core3)
        mode3_features = torch.mean(mode3_features, dim=(1, 2))  # (B*C, rank)
        
        # Concatenate mode features
        tt_features = torch.cat([mode1_features, mode2_features, mode3_features], dim=1)  # (B*C, 3*rank)
        
        # Reshape back to batch format
        tt_features = tt_features.view(batch_size, channels * self.output_features)  # (B, C*3*rank)
        
        # Final projection
        output = self.feature_projection(tt_features)
        
        return output


###############################################################################
# Global Attention Module
###############################################################################
class GlobalAttention3D(nn.Module):
    """
    TorchScript-compatible global attention module for 3D feature maps.
    Uses manual attention implementation instead of nn.MultiheadAttention.
    """
    
    def __init__(self, in_channels=64, embed_dim=128, output_dim=64, num_heads=2):
        super(GlobalAttention3D, self).__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.output_dim = output_dim
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Map from input channels to embedding dimension
        self.channel_proj = nn.Linear(in_channels, embed_dim)
        
        # Manual attention projections (instead of nn.MultiheadAttention)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Map from embedding dimension to output channels
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
        # Learnable positional encoding will be initialized based on input size
        self.pos_encoding = None
        self.scale = self.head_dim ** -0.5
        
    def _init_positional_encoding(self, D, H, W, device):
        """Initialize learnable positional encoding for DxHxW spatial positions"""
        num_positions = D * H * W
        pos_encoding = torch.randn(1, num_positions, self.embed_dim, device=device)
        nn.init.normal_(pos_encoding, std=0.02)
        self.pos_encoding = nn.Parameter(pos_encoding)
        
    def forward(self, x):
        """
        x: (batch_size, channels, D, H, W)
        Returns: (batch_size, output_dim, D, H, W) -> reshaped back to conv format
        """
        batch_size, channels, D, H, W = x.shape
        seq_len = D * H * W
        
        # Initialize positional encoding on first forward pass
        if self.pos_encoding is None:
            self._init_positional_encoding(D, H, W, x.device)
        
        # Reshape to treat spatial positions as sequence tokens
        # (batch_size, channels, D, H, W) -> (batch_size, D*H*W, channels)
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous()  # (batch, D, H, W, channels)
        x_flat = x_flat.view(batch_size, seq_len, channels)  # (batch, seq_len, channels)
        
        # Project to embedding dimension
        x_flat = self.channel_proj(x_flat)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        x_flat = x_flat + self.pos_encoding  # (batch, seq_len, embed_dim)
        
        # Manual multi-head attention
        # Project to Q, K, V
        q = self.q_proj(x_flat)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x_flat)  # (batch, seq_len, embed_dim)
        v = self.v_proj(x_flat)  # (batch, seq_len, embed_dim)
        
        # Reshape for multi-head attention
        # (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len) -> (batch, num_heads, seq_len, seq_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)  # (batch, seq_len, embed_dim)
        
        # Project to output channels per token
        output = self.output_proj(attn_output)  # (batch, seq_len, output_dim)
        
        # Reshape back to conv format: (batch, seq_len, output_dim) -> (batch, output_dim, D, H, W)
        output = output.permute(0, 2, 1)  # (batch, output_dim, seq_len)
        output = output.view(batch_size, self.output_dim, D, H, W)  # (batch, output_dim, D, H, W)
        
        return output


###############################################################################
# 3D Residual Block
###############################################################################
class ResidualBlock3D(nn.Module):
    """
    A 3D adaptation of a 2-layer residual block.
    Uses (3,3,3) kernels and a skip connection.
    'stride' can be a tuple (strideD, strideH, strideW).
    Includes post skip-connection processing layer.
    """

    def __init__(self, in_channels, out_channels, stride=(1, 2, 2)):
        super(ResidualBlock3D, self).__init__()

        # First 3D conv (no padding - manually applied in _apply_circular_padding)
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        # Second 3D conv (stride=1 here, no padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Post skip-connection processing layer (no padding)
        self.conv_post = nn.Conv3d(out_channels, out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=0,
                                   bias=False)
        self.bn_post = nn.BatchNorm3d(out_channels)

        self.activation = nn.GELU()

        # Shortcut for matching dimensions if channel or stride changes
        self.shortcut = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def _apply_circular_padding(self, x):
        """
        Apply circular padding for width axis (circumferential) and regular padding for time and height.
        For 3x3x3 kernel, we need padding=1 on all sides.
        x shape: (batch, channels, time, height, width)
        """
        # By convention, padding tuple is (pad_last_dim, pad_last_dim, pad_2nd_last, pad_2nd_last, ...)
        # We want: width=circular(1,1), height=constant(1,1), time=constant(1,1)
        
        # First circular padding only to width dimension
        x = torch.nn.functional.pad(x, (1, 1, 0, 0, 0, 0), mode='circular')
        # Then constant padding to time and height dimensions, other dimensions remain unchanged
        x = torch.nn.functional.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0)

        return x

    def forward(self, x):
        # Main path
        out = self._apply_circular_padding(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)

        out = self._apply_circular_padding(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut
        if self.shortcut is not None:
            x = self.shortcut(x)

        # Residual connection
        out += x

        # Post skip-connection processing
        out = self._apply_circular_padding(out)
        out = self.conv_post(out)
        out = self.bn_post(out)
        out = self.activation(out)
        
        return out


###############################################################################
# PetNetImproved3D with Tensor Train Decomposition
###############################################################################
class PetNetImproved3D(nn.Module):
    """
    A 3D version of PetNet:
     - Expects input of shape (batch_size, 2, T, 496, 84)
       i.e. 2 "channels" (inner, outer), T frames, 496 Height x 84 Width spatial.
       Width refers to the circumferencial axis. 
     - Uses 3D residual blocks.
     - Global attention applied after layer2 (at higher dimensionality).
     - Uses Tensor Train decomposition for spatial feature compression.
     - Output => 'num_classes' coordinates or labels.
    """

    def __init__(self, num_classes=6, tt_rank=32):
        print("Loading PetnetImproved3D Model with Tensor Train Decomposition...")
        super(PetNetImproved3D, self).__init__()

        self.tt_rank = tt_rank

        # Initial 3D conv: 2 => 16 channels
        # kernel_size=3 => (3,3,3)
        # stride=(1,1,1) so we do not reduce T,H,W here
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=16,
                                 kernel_size=3, stride=(1, 1, 1),
                                 padding=1, bias=False)
        self.bn_in = nn.BatchNorm3d(16)
        self.activation = nn.GELU()

        # First two residual blocks
        self.layer1 = ResidualBlock3D(16, 32, stride=(1, 2, 2))  # downsample H,W
        self.layer2 = ResidualBlock3D(32, 64, stride=(1, 2, 2))  # downsample

        # Global attention after layer2 (64 channels, higher spatial resolution)
        self.global_attention = GlobalAttention3D(
            in_channels=64, 
            embed_dim=128, 
            output_dim=64, 
            num_heads=8
        )

        # Remaining residual blocks (continuing from 64 channels after TT)
        self.layer3 = ResidualBlock3D(64, 128, stride=(1, 2, 2))  # downsample
        self.layer4 = ResidualBlock3D(128, 256, stride=(1, 2, 2))  # downsample
        # Removed layer5 - apply tensor train directly after layer4

        # Tensor Train decomposition (will be initialized after first forward pass)
        self.tensor_train = None
        
        self.dropout = nn.Dropout(0.3)

        # We'll compute fc_in_features dynamically after TT + remaining layers
        fc_in_features = self._compute_fc_input_size()
        self.fc1 = nn.Linear(fc_in_features, 1024, bias=True)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

        self._initialize_weights()

    def _compute_fc_input_size(self, C=2, T=3, H=207, W=41):
        """
        Runs a dummy forward pass with shape (1, 2, T, H, W)
        to determine final flattened size going into the FC layer.
        Now applies TT right after global attention.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, C, T, H, W)  # (N=1, C=2, D=T, H=207, W=41)
            out = self.conv_in(dummy)
            out = self.bn_in(out)
            out = self.activation(out)

            out = self.layer1(out)
            out = self.layer2(out)
            
            # Apply attention after layer2
            out = self.global_attention(out)
            
            # Initialize and apply tensor train right after attention
            if self.tensor_train is None:
                feature_shape = out.shape[1:]  # (channels, D, H, W)
                self.tensor_train = OptimizedTensorTrain3D(feature_shape, rank=self.tt_rank)
            
            # Apply TT to get intermediate representation
            tt_features = self.tensor_train(out)  # => (B, 1024)
            
            # Continue with remaining conv layers
            out = self.layer3(out)
            out = self.layer4(out)

            # Global average pooling for final conv features
            conv_features = torch.mean(out, dim=(2, 3, 4))  # => (B, 256)
            
            # Combine TT features and conv features
            total_features = tt_features.shape[1] + conv_features.shape[1]  # 1024 + 256 = 1280
            return total_features

    def forward(self, x, debug=False):
        """
        x: expected shape (batch_size, 2, T, 496, 84)
           (i.e. 2 channels, T frames, 496x84 spatial)
        """
        # initial conv
        if debug: print(f"{x.shape} Input shape")
        x = self.conv_in(x)  # => (B,16,T,496,84)
        if debug: print(f"{x.shape} After conv_in")
        x = self.bn_in(x)
        if debug: print(f"{x.shape} After bn_input")
        x = self.activation(x)
        if debug: print(f"{x.shape} After activation")

        # first two residual blocks
        x = self.layer1(x)  # => (B,32,T, 496/2=248, 84/2=42), etc.
        if debug: print(f"{x.shape} After layer 1")
        x = self.layer2(x)
        if debug: print(f"{x.shape} After layer 2")

        # global attention after layer2 (at higher dimensionality)
        x = self.global_attention(x)  # => (B, 64, D, H, W)
        if debug: print(f"{x.shape} After global attention")
        
        # Ensure tensor is contiguous before tensor train
        x = x.contiguous()

        # Initialize tensor train on first forward pass
        if self.tensor_train is None:
            # Get feature shape excluding batch dimension
            feature_shape = x.shape[1:]  # (channels, D, H, W)
            print(f"Initializing Tensor Train with input shape: {feature_shape}")
            self.tensor_train = OptimizedTensorTrain3D(feature_shape, rank=self.tt_rank)
            if next(self.parameters()).is_cuda:
                self.tensor_train = self.tensor_train.cuda()

        # Apply tensor train decomposition right after attention
        tt_features = self.tensor_train(x)  # => (B, 1024)
        if debug: print(f"{tt_features.shape} After tensor train decomposition")

        # Continue with remaining conv layers using original spatial features
        x = self.layer3(x)
        if debug: print(f"{x.shape} After layer 3")
        x = self.layer4(x)
        if debug: print(f"{x.shape} After layer 4")

        # Global average pooling for final conv features
        conv_features = torch.mean(x, dim=(2, 3, 4))  # => (B, 256)
        if debug: print(f"{conv_features.shape} After conv global pooling")

        # Combine TT features and conv features
        x = torch.cat([tt_features, conv_features], dim=1)  # => (B, 1024 + 256 = 1280)
        if debug: print(f"{x.shape} After combining TT and conv features")

        # FC layers
        x = self.fc1(x)
        if debug: print(f"{x.shape} After fc layer 1")
        x = self.activation(x)
        x = self.dropout(x)
        if debug: print(f"{x.shape} After activation and dropout")
        x = self.fc2(x)
        if debug: print(f"{x.shape} After fc layer 2 (output)")

        return x

    def _initialize_weights(self):
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

    # Model instantiation
    model = PetNetImproved3D(num_classes=CLASSES, tt_rank=8).to(device)
    
    # Test a dummy pass
    dummy_input = torch.randn(B, C, T, H, W).to(device)
    dummy_target = torch.randn(B, CLASSES).to(device)
    model.forward(dummy_input, debug=True)
    
    # Print parameter count after initialization
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,}")

    # Dummy training loop to observe loss reduction
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    import time

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