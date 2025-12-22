# save as: verify_vitcod_model.py
import torch
import numpy as np
import os

# Paths
checkpoint_path = './exp/lowrank_sparse/tiny_info90/checkpoint_best.pth'
mask_path = './masks/tiny/info_0.9.npy'
output_dir = './hardware_weights'

os.makedirs(output_dir, exist_ok=True)

# Load checkpoint
print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model_state = checkpoint['model']

print(f"\n{'='*60}")
print("MODEL WEIGHTS ANALYSIS")
print(f"{'='*60}")

# Check for encoder/decoder weights
encoder_decoder_found = False
for i in range(12):
    prefix = f'blocks.{i}.attn.'
    
    # Check if encoder/decoder exist
    enc_q_key = f'{prefix}encoder_q.weight'
    dec_q_key = f'{prefix}decoder_q.weight'
    enc_k_key = f'{prefix}encoder_k.weight'
    dec_k_key = f'{prefix}decoder_k.weight'
    
    if enc_q_key in model_state:
        encoder_decoder_found = True
        enc_q = model_state[enc_q_key]
        dec_q = model_state[dec_q_key]
        enc_k = model_state[enc_k_key]
        dec_k = model_state[dec_k_key]
        
        print(f"\nBlock {i}:")
        print(f"  encoder_q: {enc_q.shape} (num_heads -> hidden)")
        print(f"  decoder_q: {dec_q.shape} (hidden -> num_heads)")
        print(f"  encoder_k: {enc_k.shape}")
        print(f"  decoder_k: {dec_k.shape}")
        
        # Save weights for hardware
        np.save(f'{output_dir}/block{i}_encoder_q_weight.npy', enc_q.numpy())
        np.save(f'{output_dir}/block{i}_encoder_q_bias.npy', model_state[f'{prefix}encoder_q.bias'].numpy())
        np.save(f'{output_dir}/block{i}_decoder_q_weight.npy', dec_q.numpy())
        np.save(f'{output_dir}/block{i}_decoder_q_bias.npy', model_state[f'{prefix}decoder_q.bias'].numpy())
        np.save(f'{output_dir}/block{i}_encoder_k_weight.npy', enc_k.numpy())
        np.save(f'{output_dir}/block{i}_encoder_k_bias.npy', model_state[f'{prefix}encoder_k.bias'].numpy())
        np.save(f'{output_dir}/block{i}_decoder_k_weight.npy', dec_k.numpy())
        np.save(f'{output_dir}/block{i}_decoder_k_bias.npy', model_state[f'{prefix}decoder_k.bias'].numpy())

if encoder_decoder_found:
    print(f"\n✅ Encoder/Decoder weights found and saved!")
else:
    print(f"\n❌ ERROR: Encoder/Decoder weights NOT found!")

# Load and analyze sparse masks
print(f"\n{'='*60}")
print("SPARSE MASKS ANALYSIS")
print(f"{'='*60}")

masks = np.load(mask_path, allow_pickle=True)
print(f"Masks shape: {masks.shape}")
print(f"Masks dtype: {masks.dtype}")

# Save masks for hardware
np.save(f'{output_dir}/sparse_masks.npy', masks)
print(f"✅ Sparse masks saved!")

# Print sparsity per layer
if len(masks.shape) > 0:
    for i, mask in enumerate(masks):
        if hasattr(mask, 'shape'):
            total = mask.size
            zeros = np.sum(mask == 0) if isinstance(mask, np.ndarray) else 0
            sparsity = zeros / total * 100 if total > 0 else 0
            print(f"  Layer {i}: shape={mask.shape}, sparsity={sparsity:.2f}%")

# List all saved files
print(f"\n{'='*60}")
print("SAVED FILES FOR HARDWARE")
print(f"{'='*60}")
for f in sorted(os.listdir(output_dir)):
    size = os.path.getsize(f'{output_dir}/{f}')
    print(f"  {f}: {size/1024:.2f} KB")

print(f"\n✅ All weights saved to: {output_dir}/")