import torch
import torch.nn as nn
from .adversarial import g_hinge
from .perceptual import PerceptualLoss
from .structural import ssim_loss
from .edge import edge_loss
from .illumination import illumination_loss

class EdgeLightLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.w=weights
        self.perc=PerceptualLoss()
    def forward(self, fake, real, logits):
        terms={}
        terms["rec"]=torch.nn.functional.l1_loss(fake,real)
        terms["adv"]=sum(g_hinge(z) for z in logits)
        terms["perc"]=self.perc(fake,real)
        terms["ssim"]=ssim_loss(fake,real)
        terms["edge"]=edge_loss(fake,real)
        terms["illum"]=illumination_loss(fake,real)
        total=sum(self.w[k]*terms[k] for k in terms)
        return total,terms
