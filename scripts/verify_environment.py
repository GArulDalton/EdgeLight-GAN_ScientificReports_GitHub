import sys,platform,torch
print("Python:",sys.version)
print("Platform:",platform.platform())
print("PyTorch:",torch.__version__)
print("CUDA:",torch.version.cuda)
print("CUDA available:",torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:",torch.cuda.get_device_name(0))
    print("GPU memory GB:",torch.cuda.get_device_properties(0).total_memory/1024**3)
