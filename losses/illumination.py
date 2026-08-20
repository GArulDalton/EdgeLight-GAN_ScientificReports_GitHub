import torch.nn.functional as F

def illumination(x):
    return x.max(dim=1,keepdim=True).values

def illumination_loss(x,y):
    return F.l1_loss(illumination(x), illumination(y))
