import torch.nn.functional as F

def ssim_loss(x, y):
    mx = F.avg_pool2d(x, 11, 1, 5)
    my = F.avg_pool2d(y, 11, 1, 5)
    vx = F.avg_pool2d(x*x, 11, 1, 5) - mx*mx
    vy = F.avg_pool2d(y*y, 11, 1, 5) - my*my
    c = 0.01**2
    d = 0.03**2
    s = ((2*mx*my+c)*(2*F.avg_pool2d(x*y,11,1,5)-2*mx*my+d) /
         ((mx*mx+my*my+c)*(vx+vy+d)+1e-8))
    return 1-s.mean()
