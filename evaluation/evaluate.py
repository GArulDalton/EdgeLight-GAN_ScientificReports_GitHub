import argparse,csv
from pathlib import Path
import numpy as np
from PIL import Image
from .metrics import psnr,ssim

def arr(p): return np.asarray(Image.open(p).convert("RGB")).astype("float32")/255

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True);ap.add_argument("--reference",required=True);ap.add_argument("--output",required=True)
    a=Path(ap.parse_args().input);b=Path(ap.parse_args().reference)
    rows=[]
    for p in sorted(a.iterdir()):
        q=b/p.name
        if q.exists():
            x,y=arr(p),arr(q)
            rows.append({"file":p.name,"PSNR":psnr(x,y),"SSIM":ssim(x,y)})
    if not rows: raise RuntimeError("No paired images found.")
    with open(ap.parse_args().output,"w",newline="") as f:
        w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    print("Mean PSNR:",np.mean([r["PSNR"] for r in rows]))
    print("Mean SSIM:",np.mean([r["SSIM"] for r in rows]))
if __name__=="__main__":main()
