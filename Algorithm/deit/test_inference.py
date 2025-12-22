# save as: test_inference.py
import torch
import sys
sys.path.insert(0, '/home/bilal-linux/Research/ViTCoD/Algorithm/deit')

from models import deit_tiny_patch16_224
from PIL import Image
from torchvision import transforms

# Load model with ViTCoD components
print("Loading ViTCoD model...")
model = deit_tiny_patch16_224(
    pretrained=False,
    svd_type='mix_head_fc_qk',
    mask_path='./masks/tiny/info_0.9.npy',
    need_weight=False
)

# Load trained weights
checkpoint = torch.load('./exp/lowrank_sparse/tiny_info90/checkpoint_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

# Create dummy input (or use real image)
dummy_input = torch.randn(1, 3, 224, 224)

# Run inference
print("Running inference...")
with torch.no_grad():
    output, recon_loss = model(dummy_input, evaluate=True)

print(f"Output shape: {output.shape}")
print(f"Predicted class: {output.argmax(dim=1).item()}")
print(f"Recon loss: {recon_loss}")
print("\n✅ Software inference working!")