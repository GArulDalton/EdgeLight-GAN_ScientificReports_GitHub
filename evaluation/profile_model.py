import yaml,torch
from models.generator import EdgeLightGenerator
from fvcore.nn import FlopCountAnalysis

cfg=yaml.safe_load(open("configs/config.yaml"))
G=EdgeLightGenerator(**cfg["model"]).eval()
x=torch.randn(1,3,256,256)
p=sum(v.numel() for v in G.parameters())
f=FlopCountAnalysis(G,x).total()
print(f"Generator parameters: {p:,} ({p/1e6:.6f} M)")
print(f"Generator FLOPs: {f/1e9:.6f} GFLOPs")
