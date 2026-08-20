import argparse,time,yaml,torch
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor
from models.generator import EdgeLightGenerator
from utils.checkpoint import load_generator

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/config.yaml");ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--input",required=True)
    args=ap.parse_args()
    c=yaml.safe_load(open(args.config)); dev="cuda" if torch.cuda.is_available() else "cpu"
    G=EdgeLightGenerator(**c["model"]).to(dev).eval();load_generator(args.checkpoint,G,dev)
    p=next(Path(args.input).glob("*"))
    x=to_tensor(Image.open(p).convert("RGB").resize((256,256))).unsqueeze(0).to(dev)
    w=c["benchmark"]["warmup_iterations"];n=c["benchmark"]["timed_iterations"]
    with torch.inference_mode():
        for _ in range(w):G(x)
        if dev=="cuda":torch.cuda.synchronize()
        t=time.perf_counter()
        for _ in range(n):G(x)
        if dev=="cuda":torch.cuda.synchronize()
    ms=(time.perf_counter()-t)*1000/n
    print(f"Latency: {ms:.4f} ms/image")
    print(f"FPS: {1000/ms:.4f}")
if __name__=="__main__":main()
