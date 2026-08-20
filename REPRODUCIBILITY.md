# Reproducibility record

The manuscript specifies:

- Intel Core i9-13900K CPU at 3.0 GHz
- 64 GB DDR5 RAM
- NVIDIA RTX 4090, 24 GB
- Ubuntu 22.04 LTS
- Python 3.11
- PyTorch 2.2
- CUDA 12.1
- cuDNN 8.9
- 256 × 256 input
- fixed seed 42
- three repeated experimental runs
- generator-only inference
- discriminator used only during adversarial training

Run:

```bash
python scripts/verify_environment.py
python scripts/verify_parameters.py
python evaluation/profile_model.py
python scripts/reproducibility_report.py
```

Do not edit generated logs to match the manuscript.
