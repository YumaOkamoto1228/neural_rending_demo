import torch
from streamdiffusion.image_utils import postprocess_image

print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU Device: CUDA not available")
