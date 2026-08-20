import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, stride, k//2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, stride, k//2, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class ResidualAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.body = nn.Sequential(
            ConvBlock(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        r = self.body(x)
        return x + r * self.gate(r)

class LearnableChannelGate(nn.Module):
    """Continuous feature-channel modulation; not hard pruning or dynamic kernels."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
    def forward(self, x):
        z = self.pool(x)
        g = torch.sigmoid(self.fc2(F.relu(self.fc1(z), inplace=True)))
        return x * (1.0 + g), g

class EdgeAwareFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse = ConvBlock(channels * 2, channels, 1)
    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                          device=x.device, dtype=x.dtype).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                          device=x.device, dtype=x.dtype).view(1,1,3,3)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-6)
        edge = edge.expand_as(x)
        return self.fuse(torch.cat([x, x * (1 + edge)], dim=1))
