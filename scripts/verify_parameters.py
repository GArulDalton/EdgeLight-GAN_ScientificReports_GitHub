import yaml
from models import EdgeLightGenerator,MultiScaleDiscriminator
c=yaml.safe_load(open("configs/config.yaml"))
G=EdgeLightGenerator(**c["model"]);D=MultiScaleDiscriminator(3,c["model"]["discriminator_base"])
gp=sum(p.numel() for p in G.parameters());dp=sum(p.numel() for p in D.parameters())
print(f"Generator: {gp:,} ({gp/1e6:.6f} M)")
print(f"Discriminator: {dp:,} ({dp/1e6:.6f} M)")
print(f"Total: {gp+dp:,} ({(gp+dp)/1e6:.6f} M)")
