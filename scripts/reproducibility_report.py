import json,platform,sys,yaml,torch
from pathlib import Path
c=yaml.safe_load(open("configs/config.yaml"))
r={"python":sys.version,"platform":platform.platform(),"pytorch":torch.__version__,
   "cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),
   "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
   "seed":c["seed"],"input_resolution":c["data"]["image_size"],
   "batch_size":c["training"]["batch_size"],"epochs":c["training"]["epochs"],
   "learning_rate":c["training"]["learning_rate"],"repeated_runs":3}
Path("outputs").mkdir(exist_ok=True)
Path("outputs/reproducibility.json").write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
