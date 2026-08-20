import torch
import torch.nn.functional as F

def edges(x):
    g=x.mean(1,keepdim=True)
    kx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],device=x.device,dtype=x.dtype).view(1,1,3,3)
    ky=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],device=x.device,dtype=x.dtype).view(1,1,3,3)
    return torch.sqrt(F.conv2d(g,kx,padding=1).square()+F.conv2d(g,ky,padding=1).square()+1e-6)

def edge_loss(x,y):
    return F.l1_loss(edges(x),edges(y))
