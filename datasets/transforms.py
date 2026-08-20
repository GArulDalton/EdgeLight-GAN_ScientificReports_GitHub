import random
import torchvision.transforms.functional as TF
from PIL import Image

def paired_transform(low, high, size=256, train=True):
    low=low.resize((size,size),Image.BICUBIC)
    high=high.resize((size,size),Image.BICUBIC)
    if train:
        if random.random()<0.5:
            low,high=TF.hflip(low),TF.hflip(high)
        if random.random()<0.5:
            low,high=TF.vflip(low),TF.vflip(high)
    return TF.to_tensor(low),TF.to_tensor(high)
