# /home/bilal-linux/Research/datasets/imagenet/train/n02102040

# save as: quick_inference.py
import torch
import torch.nn.functional as F
import sys
from PIL import Image
from torchvision import transforms
import urllib.request
import os

sys.path.insert(0, '/home/bilal-linux/Research/ViTCoD/Algorithm/deit')
from models import deit_tiny_patch16_224

# Download labels
labels_path = '/tmp/imagenet_classes.txt'
if not os.path.exists(labels_path):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", 
        labels_path
    )
with open(labels_path) as f:
    labels = [line.strip() for line in f.readlines()]

# Load model
print("Loading ViTCoD model...")
model = deit_tiny_patch16_224(
    pretrained=False,
    svd_type='mix_head_fc_qk',
    mask_path='./masks/tiny/info_0.9.npy',
    need_weight=False
)
checkpoint = torch.load('./exp/lowrank_sparse/tiny_info90/checkpoint_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

# Get image path from command line or use default
image_path = sys.argv[1] if len(sys.argv) > 1 else None

if image_path and os.path.exists(image_path):
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
else:
    print("No image provided, using random tensor...")
    input_tensor = torch.randn(1, 3, 224, 224)
    image_path = "random_tensor"

# Inference
with torch.no_grad():
    output, recon_loss = model(input_tensor, evaluate=True)

# Results
probs = F.softmax(output[0], dim=0)
top5_probs, top5_indices = torch.topk(probs, 5)

print(f"\n{'='*50}")
print(f"Image: {image_path}")
print(f"Reconstruction Loss: {recon_loss.item():.4f}")
print(f"{'='*50}")
print(f"\nTop-5 Predictions:")
print(f"{'-'*50}")
for i in range(5):
    idx = top5_indices[i].item()
    prob = top5_probs[i].item() * 100
    print(f"  {i+1}. {labels[idx]:<30} {prob:.2f}%")
print(f"{'-'*50}")
print(f"\n🎯 PREDICTION: {labels[top5_indices[0].item()]} ({top5_probs[0].item()*100:.2f}%)\n")