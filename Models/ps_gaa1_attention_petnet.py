# takes attention_petnet.py and places the global attention layer after layer 2
# for higher dimensional G.A.

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
        
    def _init_positional_encoding(self, D, H, W):
        """Initialize learnable positional encoding for DxHxW spatial positions"""
        num_positions = D * H * W
        self.pos_encoding = nn.Parameter(torch.randn(1, num_positions, self.embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)
        
    def forward(self, x):
        """
        x: (batch_size, channels, D, H, W)
        Returns: (batch_size, output_dim, D, H, W) -> reshaped back to conv format
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
    Residual block with 3D depthwise + pointwise convolution.
    Uses circular padding on width, constant elsewhere.
    Includes post skip-connection processing layer.
    """
    def __init__(self, in_channels, out_channels, stride=(1, 2, 2)):
        super(ResidualBlock3D, self).__init__()

        # Depthwise 3D convolution (groups=in_channels)
        self.dw_conv1 = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=0,
            groups=in_channels, bias=False
        )
        self.pw_conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)

        # Depthwise 3D convolution (stride=1)
        self.dw_conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=0,
            groups=out_channels, bias=False
        )
        self.pw_conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Post skip-connection processing layer (depthwise-pointwise)
        self.dw_conv_post = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=0,
            groups=out_channels, bias=False
        )
        self.pw_conv_post = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_post = nn.BatchNorm3d(out_channels)

        self.activation = nn.GELU()

        # Shortcut for matching dimensions
        self.shortcut = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def _apply_circular_padding(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 0, 0, 0, 0), mode='circular')
        x = torch.nn.functional.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
        return x

    def forward(self, x):
        # Main path
        out = self._apply_circular_padding(x)
        out = self.dw_conv1(out)
        out = self.pw_conv1(out)
        out = self.bn1(out)
        out = self.activation(out)

        out = self._apply_circular_padding(out)
        out = self.dw_conv2(out)
        out = self.pw_conv2(out)
        out = self.bn2(out)

        # Shortcut
        if self.shortcut is not None:
            x = self.shortcut(x)

        out += x

        out = self._apply_circular_padding(out)
        out = self.dw_conv_post(out)
        out = self.pw_conv_post(out)
        out = self.bn_post(out)
        out = self.activation(out)

        return out



###############################################################################
# PetNetImproved3D with Early Global Attention
###############################################################################
class PetNetImproved3D(nn.Module):
    def __init__(self, num_classes=6):
        print("Loading PetnetImproved3D Model with Early Global Attention...")
        super(PetNetImproved3D, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv3d(2, 2, kernel_size=3, stride=1, padding=1, groups=2, bias=False),    # depthwise
            nn.Conv3d(2, 16, kernel_size=1, stride=1, padding=0, bias=False)              # pointwise
        )
        self.bn_in = nn.BatchNorm3d(16)

        self.bn_in = nn.BatchNorm3d(16)
        self.activation = nn.GELU()

        self.layer1 = ResidualBlock3D(16, 32, stride=(1, 2, 2))      # downsample H,W

        # Global attention after layer1 (32 channels)
        self.global_attention = GlobalAttention3D(
            in_channels=32,
            embed_dim=128,
            output_dim=32,     # must match "out_channels"
            num_heads=8
        )

        self.layer2 = ResidualBlock3D(32, 64, stride=(1, 2, 2))
        self.layer3 = ResidualBlock3D(64, 128, stride=(1, 2, 2))
        self.layer4 = ResidualBlock3D(128, 256, stride=(1, 2, 2))
        self.layer5 = ResidualBlock3D(256, 512, stride=(1, 2, 2))

        self.dropout = nn.Dropout(0.3)

        fc_in_features = self._compute_fc_input_size()
        self.fc1 = nn.Linear(fc_in_features, 1024, bias=True)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

        self._initialize_weights()

    def _compute_fc_input_size(self, C=2, T=3, H=207, W=41):
        with torch.no_grad():
            dummy = torch.zeros(1, C, T, H, W)
            out = self.conv_in(dummy)
            out = self.bn_in(out)
            out = self.activation(out)
            out = self.layer1(out)
            # Global attention after layer1
            out = self.global_attention(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = torch.mean(out, dim=(2, 3, 4))
            return out.shape[1]

    def forward(self, x, debug=False):
        if debug: print(f"{x.shape} Input shape")
        x = self.conv_in(x)
        if debug: print(f"{x.shape} After conv_in")
        x = self.bn_in(x)
        if debug: print(f"{x.shape} After bn_input")
        x = self.activation(x)
        if debug: print(f"{x.shape} After activation")

        x = self.layer1(x)
        if debug: print(f"{x.shape} After layer 1")

        # Move attention here
        x = self.global_attention(x)
        if debug: print(f"{x.shape} After global attention (now after layer 1)")

        x = self.layer2(x)
        if debug: print(f"{x.shape} After layer 2")
        x = self.layer3(x)
        if debug: print(f"{x.shape} After layer 3")
        x = self.layer4(x)
        if debug: print(f"{x.shape} After layer 4")
        x = self.layer5(x)
        if debug: print(f"{x.shape} After layer 5")

        x = torch.mean(x, dim=(2, 3, 4))
        if debug: print(f"{x.shape} After global average pooling")
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