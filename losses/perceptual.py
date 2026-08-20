import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        for p in net.parameters():
            p.requires_grad = False
        self.net = net.eval()
    def forward(self, x, y):
        return torch.nn.functional.l1_loss(self.net(x), self.net(y))
