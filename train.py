import argparse,yaml,torch
from torch.utils.data import DataLoader
from models import EdgeLightGenerator,MultiScaleDiscriminator
from datasets import PairedLowLightDataset
from losses.adversarial import d_hinge
from losses.total import EdgeLightLoss
from utils.seed import set_seed
from utils.checkpoint import save
from utils.logger import Logger

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/config.yaml");a=ap.parse_args()
    c=yaml.safe_load(open(a.config));set_seed(c["seed"]);dev="cuda" if torch.cuda.is_available() else "cpu"
    d,t,m=c["data"],c["training"],c["model"]
    ds=PairedLowLightDataset(d["train_low"],d["train_high"],256,True)
    dl=DataLoader(ds,batch_size=t["batch_size"],shuffle=True,num_workers=t["num_workers"],pin_memory=True)
    G=EdgeLightGenerator(**m).to(dev);D=MultiScaleDiscriminator(3,m["discriminator_base"]).to(dev)
    oG=torch.optim.Adam(G.parameters(),lr=t["learning_rate"],betas=tuple(t["betas"]))
    oD=torch.optim.Adam(D.parameters(),lr=t["learning_rate"],betas=tuple(t["betas"]))
    sG=torch.optim.lr_scheduler.CosineAnnealingLR(oG,t["epochs"]);sD=torch.optim.lr_scheduler.CosineAnnealingLR(oD,t["epochs"])
    w={k.split("lambda_")[1]:v for k,v in c["loss"].items()}
    loss=EdgeLightLoss(w).to(dev);log=Logger("outputs/train.csv")
    for ep in range(1,t["epochs"]+1):
        G.train();D.train();total=0
        for b in dl:
            low,high=b["low"].to(dev),b["high"].to(dev)
            with torch.no_grad(): fake=G(low)
            real_logits=D(high);fake_logits=D(fake)
            ld=sum(d_hinge(r,f) for r,f in zip(real_logits,fake_logits))
            oD.zero_grad();ld.backward();oD.step()
            fake=G(low);fake_logits=D(fake)
            lg,parts=loss(fake,high,fake_logits)
            oG.zero_grad();lg.backward();oG.step();total+=float(lg.detach())
        sG.step();sD.step();log.write({"epoch":ep,"loss":total/max(len(dl),1)})
        save("outputs/checkpoints/best_generator.pt",G,D,oG,oD,ep)
        print(ep,total/max(len(dl),1))
if __name__=="__main__":main()
