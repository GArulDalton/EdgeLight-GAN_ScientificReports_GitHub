import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def psnr(a,b): return float(peak_signal_noise_ratio(b,a,data_range=1.0))
def ssim(a,b): return float(structural_similarity(b,a,channel_axis=-1,data_range=1.0))

def advanced_metrics():
    """Optional LPIPS/NIQE/BRISQUE via pyiqa."""
    import torch, pyiqa
    dev="cuda" if torch.cuda.is_available() else "cpu"
    return {n:pyiqa.create_metric(n,device=dev) for n in ("lpips","niqe","brisque")}
