from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from .transforms import paired_transform

class PairedLowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir, size=256, train=True):
        self.low=Path(low_dir); self.high=Path(high_dir)
        self.names=sorted(p.name for p in self.low.iterdir()
                          if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"})
        self.size=size; self.train=train
    def __len__(self): return len(self.names)
    def __getitem__(self,i):
        n=self.names[i]
        a=Image.open(self.low/n).convert("RGB")
        b=Image.open(self.high/n).convert("RGB")
        a,b=paired_transform(a,b,self.size,self.train)
        return {"low":a.clamp(0,1),"high":b.clamp(0,1),"name":n}
