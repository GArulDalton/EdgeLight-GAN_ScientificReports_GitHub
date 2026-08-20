import torch
import torch.nn as nn
from .blocks import ConvBlock, DepthwiseSeparableConv, ResidualAttention, LearnableChannelGate, EdgeAwareFusion

class EdgeLightGenerator(nn.Module):
    """
    Manuscript-aligned generator architecture.
    The discriminator is intentionally absent from this inference model.
    """
    def __init__(self, in_channels=3, base_channels=32, branch_channels=64,
                 fusion_channels=128, compressed_channels=96,
                 residual_blocks=3, **kwargs):
        super().__init__()
        self.input = ConvBlock(in_channels, base_channels)
        self.fine = DepthwiseSeparableConv(base_channels, branch_channels)
        self.medium = ConvBlock(base_channels, branch_channels, 5)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels, branch_channels, 1),
            nn.ReLU(inplace=True)
        )
        self.fusion = ConvBlock(branch_channels * 3, fusion_channels, 1)
        self.encoder = DepthwiseSeparableConv(fusion_channels, fusion_channels, 3, 2)
        self.residual = nn.Sequential(
            *[ResidualAttention(fusion_channels) for _ in range(residual_blocks)]
        )
        self.edge_fusion = EdgeAwareFusion(fusion_channels)
        self.compress = nn.Conv2d(fusion_channels, compressed_channels, 1)
        self.channel_gate = LearnableChannelGate(compressed_channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(compressed_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, return_aux=False):
        f0 = self.input(x)
        fine = self.fine(f0)
        medium = self.medium(f0)
        global_f = self.global_branch(f0).expand(-1, -1, f0.shape[-2], f0.shape[-1])
        f = self.fusion(torch.cat([fine, medium, global_f], dim=1))
        f = self.encoder(f)
        f = self.residual(f)
        f = self.edge_fusion(f)
        f = self.compress(f)
        f, gate = self.channel_gate(f)
        y = self.decoder(f)
        if return_aux:
            return y, {"gate": gate}
        return y
