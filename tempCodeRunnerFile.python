import torch
import torch.nn as nn
import torch.nn.functional as F
import time

###############################################################################
# Global Attention Module with Skip Connection
###############################################################################
class GlobalAttention3D(nn.Module):
    """
    TorchScript-compatible global attention module for 3D feature maps with skip connection.
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
        
        # Manual attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Map from embedding dimension to output channels
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
        # Skip connection projection (if input/output channels differ)
        self.skip_proj = None
        if in_channels != output_dim:
            self.skip_proj = nn.Conv3d(in_channels, output_dim, kernel_size=1, bias=False)
            self.skip_bn = nn.BatchNorm3d(output_dim)
        
        # Learnable positional encoding will be initialized based on input size
        self.pos_encoding = None
        self.scale = self.head_dim ** -0.5
        
        # Optional: learnable gate for skip connection strength
        self.gate = nn.Parameter(torch.tensor(0.1))
        
    def _init_positional_encoding(self, D, H, W, device):
        """Initialize learnable positional encoding for DxHxW spatial positions"""
        num_positions = D * H * W
        pos_encoding = torch.randn(1, num_positions, self.embed_dim, device=device)
        nn.init.normal_(pos_encoding, std=0.02)
        self.pos_encoding = nn.Parameter(pos_encoding)
        
    def forward(self, x):
        """
        x: (batch_size, channels, D, H, W)
        Returns: (batch_size, output_dim, D, H, W)
        """
        batch_size, channels, D, H, W = x.shape
        seq_len = D * H * W
        
        # Store input for skip connection
        skip_input = x
        
        # Initialize positional encoding on first forward pass
        if self.pos_encoding is None:
            self._init_positional_encoding(D, H, W, x.device)
        
        # Reshape to treat spatial positions as sequence tokens
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous()
        x_flat = x_flat.view(batch_size, seq_len, channels)
        
        # Project to embedding dimension
        x_flat = self.channel_proj(x_flat)
        
        # Add positional encoding
        x_flat = x_flat + self.pos_encoding
        
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
        
        # Skip connection
        if self.skip_proj is not None:
            # Project skip input to match output dimensions
            skip_input = self.skip_proj(skip_input)
            skip_input = self.skip_bn(skip_input)
        
        # Add skip connection with learnable gate
        output = output + self.gate * skip_input
        
        return output


###############################################################################
# Normalized Tensor Train Module
###############################################################################
class NormalizedDirectTensorTrain3D(nn.Module):
    """
    Direct tensor train with pre-normalization for stability.
    """
    
    def __init__(self, input_shape, rank=64, norm_type='batch'):
        super(NormalizedDirectTensorTrain3D, self).__init__()
        
        self.channels, self.D, self.H, self.W = input_shape
        self.rank = rank
        self.norm_type = norm_type
        
        # Add normalization before TT decomposition
        if norm_type == 'batch':
            self.norm = nn.BatchNorm3d(self.channels)
        elif norm_type == 'layer':
            # LayerNorm over spatial dimensions
            self.norm = nn.LayerNorm([self.D, self.H, self.W])
        elif norm_type == 'group':
            # GroupNorm with 8 groups (adjust based on channels)
            num_groups = min(8, self.channels)
            self.norm = nn.GroupNorm(num_groups, self.channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm3d(self.channels)
        else:
            self.norm = nn.Identity()
        
        # TT cores with improved initialization
        self.core1 = nn.Parameter(torch.randn(1, self.D, rank) * (2.0 / (self.D + rank)) ** 0.5)
        self.core2 = nn.Parameter(torch.randn(rank, self.H, rank) * (2.0 / (self.H + 2*rank)) ** 0.5)
        self.core3 = nn.Parameter(torch.randn(rank, self.W, 1) * (2.0 / (self.W + rank)) ** 0.5)
        
        # Activation function for output stabilization
        self.activation = nn.GELU()  # Consistent with rest of architecture
        
        self.output_size = rank * 3
        
    def forward(self, x):
        """
        Apply normalization then tensor train decomposition.
        """
        batch_size, channels, D, H, W = x.shape
        
        # Normalize input for stability
        if self.norm_type == 'layer':
            # LayerNorm expects (batch, ..., normalized_dims)
            x_norm = x.permute(0, 1, 2, 3, 4)  # Keep as is for 3D LayerNorm
            x_norm = self.norm(x_norm)
        else:
            x_norm = self.norm(x)
        
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
# 3D Residual Block
###############################################################################
class ResidualBlock3D(nn.Module):
    """
    A 3D adaptation of a 2-layer residual block.
    Uses (3,3,3) kernels and a skip connection.
    """

    def __init__(self, in_channels, out_channels, stride=(1, 2, 2)):
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

        # Shortcut for matching dimensions
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
        Apply circular padding for width axis and regular padding for time and height.
        """
        # Circular padding for width, constant for others
        x = torch.nn.functional.pad(x, (1, 1, 0, 0, 0, 0), mode='circular')
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
# PetNetImproved3D with Skip Connection in Global Attention
###############################################################################
class PetNetImproved3D(nn.Module):
    """
    PetNet with normalized tensor train decomposition and skip connection in global attention.
    """

    def __init__(self, num_classes=6, tt_rank=32, norm_type='batch'):
        print(f"Loading PetnetImproved3D Model with Normalized Tensor Train (norm_type={norm_type}) and Skip Connection in Global Attention...")
        super(PetNetImproved3D, self).__init__()

        self.tt_rank = tt_rank
        self.norm_type = norm_type

        # Initial 3D conv: 2 => 16 channels
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=16,
                                 kernel_size=3, stride=(1, 1, 1),
                                 padding=1, bias=False)
        self.bn_in = nn.BatchNorm3d(16)
        self.activation = nn.GELU()

        # Residual blocks
        self.layer1 = ResidualBlock3D(16, 32, stride=(1, 2, 2))
        self.layer2 = ResidualBlock3D(32, 64, stride=(1, 2, 2))

        # Global attention after layer2 (64 input channels, 64 output channels for clean skip)
        self.global_attention = GlobalAttention3D(
            in_channels=64, 
            embed_dim=128, 
            output_dim=64,  # Keep same as input for clean skip connection
            num_heads=8
        )

        # Remaining residual blocks
        self.layer3 = ResidualBlock3D(64, 128, stride=(1, 2, 2))
        self.layer4 = ResidualBlock3D(128, 256, stride=(1, 2, 2))

        # Normalized Tensor Train decomposition (initialized dynamically)
        self.tensor_train = None
        
        self.dropout = nn.Dropout(0.3)

        # Compute FC input features dynamically
        fc_in_features = self._compute_fc_input_size()
        self.fc1 = nn.Linear(fc_in_features, 1024, bias=True)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

        self._initialize_weights()

    def _compute_fc_input_size(self, C=2, T=3, H=207, W=41):
        """
        Runs a dummy forward pass to determine TT output size.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, C, T, H, W)
            out = self.conv_in(dummy)
            out = self.bn_in(out)
            out = self.activation(out)

            out = self.layer1(out)
            out = self.layer2(out)
            
            # Apply attention after layer2 (now with skip connection)
            out = self.global_attention(out)
            
            out = self.layer3(out)
            out = self.layer4(out)
            
            # Initialize normalized tensor train
            if self.tensor_train is None:
                feature_shape = out.shape[1:]  # (channels, D, H, W)
                self.tensor_train = NormalizedDirectTensorTrain3D(
                    feature_shape, 
                    rank=self.tt_rank,
                    norm_type=self.norm_type
                )
            
            # Get TT output size
            tt_output = self.tensor_train(out)
            return tt_output.shape[1]

    def forward(self, x, debug=False):
        """
        x: expected shape (batch_size, 2, T, 496, 84)
        """
        # Initial conv
        if debug: print(f"{x.shape} Input shape")
        x = self.conv_in(x)
        if debug: print(f"{x.shape} After conv_in")
        x = self.bn_in(x)
        if debug: print(f"{x.shape} After bn_input")
        x = self.activation(x)
        if debug: print(f"{x.shape} After activation")

        # Residual blocks
        x = self.layer1(x)
        if debug: print(f"{x.shape} After layer 1")
        x = self.layer2(x)
        if debug: print(f"{x.shape} After layer 2")

        # Global attention with skip connection
        x = self.global_attention(x)
        if debug: print(f"{x.shape} After global attention (with skip)")
        
        x = self.layer3(x)
        if debug: print(f"{x.shape} After layer 3")
        x = self.layer4(x)
        if debug: print(f"{x.shape} After layer 4")
        
        # Ensure tensor is contiguous before tensor train
        x = x.contiguous()

        # Initialize normalized tensor train on first forward pass
        if self.tensor_train is None:
            feature_shape = x.shape[1:]  # (channels, D, H, W)
            print(f"Initializing Normalized Tensor Train with input shape: {feature_shape}")
            self.tensor_train = NormalizedDirectTensorTrain3D(
                feature_shape, 
                rank=self.tt_rank,
                norm_type=self.norm_type
            )
            if next(self.parameters()).is_cuda:
                self.tensor_train = self.tensor_train.cuda()

        # Apply normalized tensor train decomposition
        x = self.tensor_train(x)
        if debug: print(f"{x.shape} After normalized tensor train decomposition")

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

    # Model instantiation with normalized TT and skip connection in attention
    model = PetNetImproved3D(num_classes=CLASSES, tt_rank=32, norm_type='batch').to(device)
    
    # Test a dummy pass
    dummy_input = torch.randn(B, C, T, H, W).to(device)
    dummy_target = torch.randn(B, CLASSES).to(device)
    model.forward(dummy_input, debug=True)
    
    # Print parameter count after initialization
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,}")

    # Dummy training loop to observe loss reduction - EXACTLY AS YOURS
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