# takes dans_pet_net.py and adds attention with a smaller embedding space,
# for global attention and feartue space reduction for the fc layers. Just always
# make sure the dummy input in _compute_fc_input_size is the shape of the real input data for proper initialisation.

import torch
import torch.nn as nn

###############################################################################
# Global Attention Module
###############################################################################
class GlobalAttention3D(nn.Module):
    """
    TorchScript-compatible global attention module for 3D feature maps.
    Uses manual attention implementation instead of nn.MultiheadAttention.
    """
    
    def __init__(self, in_channels=512, embed_dim=128, output_dim=16, num_heads=2):
        super(GlobalAttention3D, self).__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
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
        
    def _init_positional_encoding(self, D, H, W):
        """Initialize learnable positional encoding for DxHxW spatial positions"""
        num_positions = D * H * W
        self.pos_encoding = nn.Parameter(torch.randn(1, num_positions, self.embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)
        
    def forward(self, x):
        """
        x: (batch_size, channels, D, H, W)
        Returns: (batch_size, D*H*W*output_dim) -> flattened for FC layers
        """
        batch_size, channels, D, H, W = x.shape
        seq_len = D * H * W
        
        # Initialize positional encoding on first forward pass
        if self.pos_encoding is None:
            self._init_positional_encoding(D, H, W)
            # Move to same device as input
            self.pos_encoding = self.pos_encoding.to(x.device)
        
        # Reshape to treat spatial positions as sequence tokens
        # (batch_size, channels, D, H, W) -> (batch_size, D*H*W, channels)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (batch, D, H, W, channels)
        x = x.view(batch_size, seq_len, channels)  # (batch, seq_len, channels)
        
        # Project to embedding dimension
        x = self.channel_proj(x)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding  # (batch, seq_len, embed_dim)
        
        # Manual multi-head attention
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)  # (batch, seq_len, embed_dim)
        v = self.v_proj(x)  # (batch, seq_len, embed_dim)
        
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
# PetNetImproved3D
###############################################################################
class PetNetImproved3D(nn.Module):
    """
    A 3D version of PetNet:
     - Expects input of shape (batch_size, 2, T, 496, 84)
       i.e. 2 "channels" (inner, outer), T frames, 496 Height x 84 Width spatial.
       Width refers to the circumferencial axis. 
     - Uses 3D residual blocks.
     - Ends with global attention.
     - Output => 'num_classes' coordinates or labels.
    """

    def __init__(self, num_classes=6):
        print("Loading PetnetImproved3D Model...")
        super(PetNetImproved3D, self).__init__()

        # Initial 3D conv: 2 => 16 channels
        # kernel_size=3 => (3,3,3)
        # stride=(1,1,1) so we do not reduce T,H,W here
        self.conv_in = nn.Conv3d(in_channels=2, out_channels=16,
                                 kernel_size=3, stride=(1, 1, 1),
                                 padding=1, bias=False)
        self.bn_in = nn.BatchNorm3d(16)
        self.activation = nn.GELU()

        # Residual blocks
        # We apply stride=(1,2,2) to reduce only in the spatial dims (H,W)
        # If you'd like to also reduce T, set strideD>1.
        self.layer1 = ResidualBlock3D(16, 32, stride=(1, 2, 2))  # downsample H,W
        self.layer2 = ResidualBlock3D(32, 64, stride=(1, 2, 2))  # downsample
        self.layer3 = ResidualBlock3D(64, 128, stride=(1, 2, 2))  # downsample
        self.layer4 = ResidualBlock3D(128, 256, stride=(1, 2, 2))  # downsample
        self.layer5 = ResidualBlock3D(256, 512, stride=(1, 2, 2))  # downsample

        # Global attention - Remove the spatial_dims parameter
        self.global_attention = GlobalAttention3D(
            in_channels=512, 
            embed_dim=256, 
            output_dim=16, 
            num_heads=16
        )

        self.dropout = nn.Dropout(0.3)

        # We'll compute fc_in_features dynamically
        fc_in_features = self._compute_fc_input_size()
        self.fc1 = nn.Linear(fc_in_features, 1024, bias=True)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

        self._initialize_weights()

    def _compute_fc_input_size(self, C=2, T=3, H=207, W=41):
        """
        Runs a dummy forward pass with shape (1, 2, T, H, W)
        to determine final flattened size going into the FC layer.

        If T is variable in real data, you might pick a "typical" T
        or the max T you expect. The final global pooling removes T anyway,
        but intermediate layers can reduce T if you used stride>1 in depth.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, C, T, H, W)  # (N=1, C=2, D=T, H=496, W=84)
            out = self.conv_in(dummy)
            out = self.bn_in(out)
            out = self.activation(out)

            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)

            out = self.global_attention(out)  # => shape (1, seq_len, output_dim)
            out = out.view(1, -1)
            return out.shape[1]

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

        # residual blocks
        x = self.layer1(x)  # => (B,32,T, 496/2=248, 84/2=42), etc.
        if debug: print(f"{x.shape} After layer 1")
        x = self.layer2(x)
        if debug: print(f"{x.shape} After layer 2")
        x = self.layer3(x)
        if debug: print(f"{x.shape} After layer 3")
        x = self.layer4(x)
        if debug: print(f"{x.shape} After layer 4")
        x = self.layer5(x)
        if debug: print(f"{x.shape} After layer 5")

        # global attention => (B, seq_len, output_dim)
        x = self.global_attention(x)
        if debug: print(f"{x.shape} After global attention")

        # flatten => (B, seq_len * output_dim)
        x = x.view(x.size(0), -1)
        if debug: print(f"{x.shape} After flattening")

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

    # B = 8
    B = 128

    C = 2
    T = 3 

    H = 207
    # H = 496

    W = 41
    # W = 84
    
    CLASSES = 6

    # Model instantiation
    model = PetNetImproved3D(num_classes=CLASSES).to(device)
    
    # Print parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,}")
    
    # Test a dummy pass
    dummy_input = torch.randn(B, C, T, H, W).to(device)
    dummy_target = torch.randn(B, CLASSES).to(device)
    model.forward(dummy_input, debug=True)

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