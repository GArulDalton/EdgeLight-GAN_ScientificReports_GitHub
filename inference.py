import argparse,yaml,torch
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from models.generator import EdgeLightGenerator
from utils.checkpoint import load_generator

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/config.yaml");ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--input",required=True);ap.add_argument("--output",required=True)
    a=ap.parse_args();c=yaml.safe_load(open(a.config));dev="cuda" if torch.cuda.is_available() else "cpu"
    G=EdgeLightGenerator(**c["model"]).to(dev).eval();load_generator(a.checkpoint,G,dev)
    Path(a.output).mkdir(parents=True,exist_ok=True)
    for p in Path(a.input).glob("*"):
        try:x=to_tensor(Image.open(p).convert("RGB").resize((256,256))).unsqueeze(0).to(dev)
        except:continue
        with torch.inference_mode():y=G(x)
        save_image(y[0].cpu(),Path(a.output)/p.name)
if __name__=="__main__":main()
