import torch.nn.functional as F

def d_hinge(real, fake):
    return 0.5 * (F.relu(1-real).mean() + F.relu(1+fake).mean())

def g_hinge(fake):
    return -fake.mean()
