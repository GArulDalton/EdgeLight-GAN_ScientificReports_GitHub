import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, channels=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*4, 1, 3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

class MultiScaleDiscriminator(nn.Module):
    """Training-only discriminator."""
    def __init__(self, channels=3, base=64):
        super().__init__()
        self.full = PatchDiscriminator(channels, base)
        self.half = PatchDiscriminator(channels, base)
    def forward(self, x):
        import torch.nn.functional as F
        return [self.full(x), self.half(F.avg_pool2d(x, 2))]
