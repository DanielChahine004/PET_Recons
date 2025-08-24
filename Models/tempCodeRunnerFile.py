# takes attention_petnet.py and places the global attention layer after layer 1
# for higher dimensional G.A. Employs point-wise seperation convolutions over
# full fat classical convolutions for an 8-10x reduction in paramater count. 
# Also introduces a residual connection after the global attention, and layer norm
# for the global attention (Apparently batch norm works better for Convolutions, 
# and layer norm works better for Transformers...we'll see about that). 
# Uses 3 full connected layers with dropout for a more regression head.   
# Modified to use windowed attention with 4x4 non-overlapping windows.

import torch
import torch.nn as nn

###############################################################################
# Windowed Global Attention Module
###############################################################################
class WindowedGlobalAttention3D(nn.Module):
    """
    Windowed global attention module for 3D feature maps using 4x4 non-overlapping windows.
    Applies attention within each window, significantly reducing computational complexity.
    """
    
    def __init__(self, in_channels=64, embed_dim=128, output_dim=64, num_heads=2, window_size=4):
        super(WindowedGlobalAttention3D, self).__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.output_dim = output_dim
        self.window_size = window_size
        
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
        
        # Learnable positional encoding for window positions
        self.pos_encoding = None
        self.scale = self.head_dim ** -0.5
        self.layernorm = nn.LayerNorm(self.embed_dim)

        
    def _init_positional_encoding(self, window_seq_len):
        """Initialize learnable positional encoding for window sequence length"""
        self.pos_encoding = nn.Parameter(torch.randn(1, window_seq_len, self.embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)
        
    def _create_windows(self, x):
        """
        Split the spatial dimensions (H, W) into non-overlapping 4x4 windows.
        Args:
            x: input tensor of shape (batch, channels, D, H, W)
        Returns:
            windows: tensor of shape (batch, num_windows, window_seq_len, channels)
            num_windows_h, num_windows_w: number of windows in H and W dimensions
        """
        batch_size, channels, D, H, W = x.shape
        
        # Calculate number of windows (pad if necessary)
        num_windows_h = (H + self.window_size - 1) // self.window_size
        num_windows_w = (W + self.window_size - 1) // self.window_size
        
        # Pad H and W to be divisible by window_size
        pad_h = num_windows_h * self.window_size - H
        pad_w = num_windows_w * self.window_size - W
        
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            H, W = H + pad_h, W + pad_w
        
        # Reshape to create windows: (batch, channels, D, num_windows_h, window_size, num_windows_w, window_size)
        x = x.view(batch_size, channels, D, num_windows_h, self.window_size, num_windows_w, self.window_size)
        
        # Permute and reshape to group windows: (batch, D, num_windows_h, num_windows_w, channels, window_size, window_size)
        x = x.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
        
        # Flatten spatial dimensions within each window and combine D with num_windows
        # Final shape: (batch, D * num_windows_h * num_windows_w, window_size * window_size, channels)
        num_windows = D * num_windows_h * num_windows_w
        window_seq_len = self.window_size * self.window_size
        windows = x.view(batch_size, num_windows, window_seq_len, channels)
        
        return windows, num_windows_h, num_windows_w, (H, W)
    
    def _restore_from_windows(self, windows, batch_size, D, num_windows_h, num_windows_w, padded_size):
        """
        Restore the original spatial structure from windowed representation.
        Args:
            windows: tensor of shape (batch, num_windows, window_seq_len, output_dim)
            Original spatial dimensions and padding info
        Returns:
            x: tensor of shape (batch, output_dim, D, H, W)
        """
        H_padded, W_padded = padded_size
        window_seq_len = self.window_size * self.window_size
        
        # Reshape back to spatial windows: (batch, D, num_windows_h, num_windows_w, output_dim, window_size, window_size)
        windows = windows.view(batch_size, D, num_windows_h, num_windows_w, self.output_dim, self.window_size, self.window_size)
        
        # Permute back: (batch, output_dim, D, num_windows_h, window_size, num_windows_w, window_size)
        windows = windows.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
        
        # Reshape to recover spatial dimensions: (batch, output_dim, D, H_padded, W_padded)
        x = windows.view(batch_size, self.output_dim, D, H_padded, W_padded)
        
        # Remove padding if it was added
        original_H = H_padded - ((num_windows_h * self.window_size) - H_padded)
        original_W = W_padded - ((num_windows_w * self.window_size) - W_padded)
        
        # Note: We need to calculate the original size differently
        # Let's just remove any padding that was added
        if H_padded > original_H or W_padded > original_W:
            # Calculate original dimensions
            orig_H = H_padded - (num_windows_h * self.window_size - H_padded)
            orig_W = W_padded - (num_windows_w * self.window_size - W_padded)
            # This calculation is complex, let's store original dims in forward pass
            pass
        
        return x
        
    def forward(self, x):
        batch_size, channels, D, H, W = x.shape
        orig_H, orig_W = H, W  # Store original dimensions
        
        # Create windows
        windows, num_windows_h, num_windows_w, padded_size = self._create_windows(x)
        num_windows, window_seq_len, _ = windows.shape[1], windows.shape[2], windows.shape[3]
        
        # Initialize positional encoding if needed
        if self.pos_encoding is None or self.pos_encoding.shape[1] != window_seq_len:
            self._init_positional_encoding(window_seq_len)
            self.pos_encoding = self.pos_encoding.to(x.device)

        # Process each window independently
        # Reshape to process all windows in batch: (batch * num_windows, window_seq_len, channels)
        windows_flat = windows.view(batch_size * num_windows, window_seq_len, channels)
        
        # Project to embedding dimension
        x_proj = self.channel_proj(windows_flat)  # (batch * num_windows, window_seq_len, embed_dim)
        residual = x_proj
        
        # Add positional encoding
        x_proj = x_proj + self.pos_encoding
        
        # Apply attention within each window
        q = self.q_proj(x_proj)
        k = self.k_proj(x_proj)
        v = self.v_proj(x_proj)
        
        # Reshape for multi-head attention
        q = q.view(batch_size * num_windows, window_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size * num_windows, window_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size * num_windows, window_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size * num_windows, window_seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection and layer norm
        attn_output = attn_output + residual
        attn_output = self.layernorm(attn_output)
        
        # Project to output dimension
        output = self.output_proj(attn_output)  # (batch * num_windows, window_seq_len, output_dim)
        
        # Reshape back to window format
        output = output.view(batch_size, num_windows, window_seq_len, self.output_dim)
        
        # Restore spatial structure
        output = self._restore_from_windows(output, batch_size, D, num_windows_h, num_windows_w, padded_size)
        
        # Remove padding to match original input size
        if output.shape[-2] != orig_H or output.shape[-1] != orig_W:
            output = output[:, :, :, :orig_H, :orig_W]
        
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
# PetNetImproved3D with Windowed Global Attention
###############################################################################
class PetNetImproved3D(nn.Module):
    def __init__(self, num_classes=6):
        print("Loading PetnetImproved3D Model with Windowed Global Attention...")
        super(PetNetImproved3D, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv3d(2, 2, kernel_size=3, stride=1, padding=1, groups=2, bias=False),    # depthwise
            nn.Conv3d(2, 16, kernel_size=1, stride=1, padding=0, bias=False)              # pointwise
        )
        self.bn_in = nn.BatchNorm3d(16)
        self.activation = nn.GELU()

        self.layer1 = ResidualBlock3D(16, 32, stride=(1, 2, 2))      # downsample H,W
        
        # Windowed global attention after layer1 (32 channels) with 4x4 windows
        self.global_attention = WindowedGlobalAttention3D(
            in_channels=32,
            embed_dim=128,
            output_dim=32,
            num_heads=8,
            window_size=4
        )

        self.layer2 = ResidualBlock3D(32, 64, stride=(1, 2, 2))
        self.layer3 = ResidualBlock3D(64, 128, stride=(1, 2, 2))
        self.layer4 = ResidualBlock3D(128, 256, stride=(1, 2, 2))
        self.layer5 = ResidualBlock3D(256, 256, stride=(1, 2, 2))

        self.dropout = nn.Dropout(0.3)

        fc_in_features = self._compute_fc_input_size()
        self.fc1 = nn.Linear(fc_in_features, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)  # final regressive output

        self._initialize_weights()

    def _compute_fc_input_size(self, C=2, T=3, H=207, W=41):
        with torch.no_grad():
            dummy = torch.zeros(1, C, T, H, W)
            out = self.conv_in(dummy)
            out = self.bn_in(out)
            out = self.activation(out)
            out = self.layer1(out)
            out = self.global_attention(out) # Windowed global attention after layer1
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = out.view(out.size(0), -1)
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

        # Windowed global attention after layer 1
        x = self.global_attention(x)
        if debug: print(f"{x.shape} After windowed global attention (after layer 1)")

        x = self.layer2(x)
        if debug: print(f"{x.shape} After layer 2")
        x = self.layer3(x)
        if debug: print(f"{x.shape} After layer 3")
        x = self.layer4(x)
        if debug: print(f"{x.shape} After layer 4")
        x = self.layer5(x)
        if debug: print(f"{x.shape} After layer 5")

        x = x.view(x.size(0), -1)  # Flatten all features (B, all_channels)
        if debug: print(f"{x.shape} After flattening all channels/voxels")
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        if debug: print(f"{x.shape} After fc layer 1, activation and dropout")
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        if debug: print(f"{x.shape} After fc layer 2 activation and dropout")
        x = self.fc3(x) 
        if debug: print(f"{x.shape} After fc layer 3 (output)")

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

    B = 1
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