import torch
import torch.nn as nn

###############################################################################
# Efficient Global Attention Module
###############################################################################
class EfficientGlobalAttention3D(nn.Module):
    """Parameter-efficient global attention using linear attention approximation"""
    
    def __init__(self, in_channels=64, embed_dim=64, num_heads=4):
        super(EfficientGlobalAttention3D, self).__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Reduce embedding dimension and heads for efficiency
        self.channel_proj = nn.Conv3d(in_channels, embed_dim, 1, bias=False)
        
        # Single conv for Q, K, V instead of separate linear layers
        self.qkv_proj = nn.Conv3d(embed_dim, embed_dim * 3, 1, bias=False)
        self.out_proj = nn.Conv3d(embed_dim, in_channels, 1, bias=False)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # Project to embedding space
        x_embed = self.channel_proj(x)  # (B, embed_dim, D, H, W)
        
        # Get Q, K, V
        qkv = self.qkv_proj(x_embed)  # (B, embed_dim*3, D, H, W)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Flatten spatial dimensions and reshape for multi-head attention
        # (B, C, D, H, W) -> (B, C, D*H*W) -> (B, heads, head_dim, D*H*W)
        spatial_size = D * H * W
        q = q.view(B, self.num_heads, self.head_dim, spatial_size)
        k = k.view(B, self.num_heads, self.head_dim, spatial_size)  
        v = v.view(B, self.num_heads, self.head_dim, spatial_size)
        
        # Standard scaled dot-product attention
        # Q: (B, heads, head_dim, spatial) K: (B, heads, head_dim, spatial)
        # QK^T: (B, heads, spatial, spatial)
        attn_weights = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        # (B, heads, spatial, spatial) @ (B, heads, spatial, head_dim) -> (B, heads, spatial, head_dim)
        out = torch.matmul(attn_weights, v.transpose(-2, -1))
        # -> (B, heads, head_dim, spatial)
        out = out.transpose(-2, -1).contiguous()
        
        # Concatenate heads and reshape back
        out = out.view(B, self.embed_dim, D, H, W)
        out = self.out_proj(out)
        
        return out + x  # Residual connection

###############################################################################
# Squeeze-and-Excitation Block
###############################################################################
class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c = x.size(0), x.size(1)
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

###############################################################################
# Depthwise Separable Convolution Block
###############################################################################
class DepthwiseSeparableConv3D(nn.Module):
    """Depthwise separable convolution for parameter efficiency"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(DepthwiseSeparableConv3D, self).__init__()
        
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

###############################################################################
# Efficient Residual Block
###############################################################################
class EfficientResidualBlock3D(nn.Module):
    """Parameter-efficient residual block using depthwise separable convs and SE"""
    
    def __init__(self, in_channels, out_channels, stride=(1, 2, 2)):
        super(EfficientResidualBlock3D, self).__init__()
        
        # First depthwise separable conv
        self.conv1 = DepthwiseSeparableConv3D(in_channels, out_channels, 3, stride)
        self.conv2 = DepthwiseSeparableConv3D(out_channels, out_channels, 3, 1)
        
        # SE block
        self.se = SEBlock3D(out_channels)
        
        self.activation = nn.GELU()
        
        # Shortcut
        self.shortcut = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def _apply_circular_padding(self, x):
        # Circular padding for width, constant for time and height
        x = torch.nn.functional.pad(x, (1, 1, 0, 0, 0, 0), mode='circular')
        x = torch.nn.functional.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
        return x
    
    def forward(self, x):
        # Main path
        out = self._apply_circular_padding(x)
        out = self.conv1(out)
        out = self.activation(out)
        
        out = self._apply_circular_padding(out)
        out = self.conv2(out)
        
        # SE attention
        out = self.se(out)
        
        # Shortcut
        if self.shortcut is not None:
            x = self.shortcut(x)
        
        out += x
        out = self.activation(out)
        
        return out

###############################################################################
# Efficient PetNet3D
###############################################################################
class EfficientPetNet3D(nn.Module):
    """Parameter-efficient version of PetNet3D"""
    
    def __init__(self, num_classes=6):
        print("Loading Efficient PetNet3D...")
        super(EfficientPetNet3D, self).__init__()
        
        # Reduced initial channels: 2 => 16
        self.conv_in = nn.Conv3d(2, 16, 3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm3d(16)
        self.activation = nn.GELU()
        
        # Efficient residual blocks with reduced channel progression
        self.layer1 = EfficientResidualBlock3D(16, 24, stride=(1, 2, 2))
        self.layer2 = EfficientResidualBlock3D(24, 32, stride=(1, 2, 2))
        
        # Efficient global attention
        self.global_attention = EfficientGlobalAttention3D(32, 32, 4)
        
        self.layer3 = EfficientResidualBlock3D(32, 64, stride=(1, 2, 2))
        self.layer4 = EfficientResidualBlock3D(64, 128, stride=(1, 2, 2))
        self.layer5 = EfficientResidualBlock3D(128, 256, stride=(1, 2, 2))
        
        self.dropout = nn.Dropout(0.3)
        
        # Compute FC input size
        fc_in_features = self._compute_fc_input_size()
        self.fc1 = nn.Linear(fc_in_features, 512, bias=True)  # Reduced from 1024
        self.fc2 = nn.Linear(512, num_classes, bias=True)
        
        self._initialize_weights()
    
    def _compute_fc_input_size(self, C=2, T=3, H=207, W=41):
        with torch.no_grad():
            dummy = torch.zeros(1, C, T, H, W)
            out = self.conv_in(dummy)
            out = self.bn_in(out)
            out = self.activation(out)
            
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.global_attention(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            
            out = torch.mean(out, dim=(2, 3, 4))
            return out.shape[1]
    
    def forward(self, x, debug=False):
        if debug: print(f"{x.shape} Input shape")
        
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.activation(x)
        if debug: print(f"{x.shape} After conv_in")
        
        x = self.layer1(x)
        if debug: print(f"{x.shape} After layer 1")
        
        x = self.layer2(x)
        if debug: print(f"{x.shape} After layer 2")
        
        x = self.global_attention(x)
        if debug: print(f"{x.shape} After global attention")
        
        x = self.layer3(x)
        if debug: print(f"{x.shape} After layer 3")
        
        x = self.layer4(x)
        if debug: print(f"{x.shape} After layer 4")
        
        x = self.layer5(x)
        if debug: print(f"{x.shape} After layer 5")
        
        x = torch.mean(x, dim=(2, 3, 4))
        if debug: print(f"{x.shape} After global average pooling")
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        if debug: print(f"{x.shape} After fc1")
        
        x = self.fc2(x)
        if debug: print(f"{x.shape} After fc2 (output)")
        
        return x
    
    def _initialize_weights(self):
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
    
    # Test parameters
    B, C, T, H, W = 8, 2, 3, 207, 41
    CLASSES = 6
    
    # Compare parameter counts
    print("\n=== Parameter Comparison ===")
    
    # Original model (commented out to avoid import issues)
    # original_model = PetNetImproved3D(num_classes=CLASSES)
    # original_params = sum(p.numel() for p in original_model.parameters())
    # print(f"Original model parameters: {original_params:,}")
    
    # Efficient model
    efficient_model = EfficientPetNet3D(num_classes=CLASSES).to(device)
    efficient_params = sum(p.numel() for p in efficient_model.parameters())
    print(f"Efficient model parameters: {efficient_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(B, C, T, H, W).to(device)
    dummy_target = torch.randn(B, CLASSES).to(device)
    
    print("\n=== Forward Pass Test ===")
    efficient_model.forward(dummy_input, debug=True)
    
    # Quick training test
    optimizer = torch.optim.Adam(efficient_model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    print("\n=== Training Test ===")
    for epoch in range(300):
        optimizer.zero_grad()
        output = efficient_model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")