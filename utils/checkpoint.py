import torch
from pathlib import Path
def save(path,G,D,optG=None,optD=None,epoch=0):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"generator":G.state_dict(),"discriminator":D.state_dict(),
                "optG":optG.state_dict() if optG else None,
                "optD":optD.state_dict() if optD else None,
                "epoch":epoch},path)
def load_generator(path,G,device):
    c=torch.load(path,map_location=device)
    G.load_state_dict(c.get("generator",c),strict=True)
