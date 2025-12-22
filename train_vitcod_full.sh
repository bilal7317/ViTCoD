#!/bin/bash
# train_vitcod_full.sh - Complete ViTCoD training pipeline
# Usage: ./train_vitcod_full.sh [tiny|small] [data_path]

set -e

# Configuration
MODEL_SIZE="${1:-tiny}"
DATA_PATH="${2:-/home/bilal-linux/Research/datasets/imagenet}"

cd /home/bilal-linux/Research/ViTCoD
# pyenv activate vitcod_py38
cd Algorithm/deit

# Set model-specific parameters
case $MODEL_SIZE in
    tiny)
        MODEL="deit_tiny_patch16_224"
        PRETRAINED="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
        BATCH_SIZE=32
        ;;
    small)
        MODEL="deit_small_patch16_224"
        PRETRAINED="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
        BATCH_SIZE=16
        ;;
esac

echo "=========================================="
echo "ViTCoD Training Pipeline"
echo "Model: $MODEL"
echo "Data: $DATA_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="

# Create directories
mkdir -p exp/lowrank/$MODEL_SIZE
mkdir -p attn/${MODEL_SIZE}_lowrank
mkdir -p masks/$MODEL_SIZE
mkdir -p exp/lowrank_sparse/${MODEL_SIZE}_info90

# STEP 1: Low-rank finetuning
echo ""
echo "===== STEP 1: Low-rank Finetuning ====="
python main.py \
    --model $MODEL \
    --resume $PRETRAINED \
    --data-path $DATA_PATH \
    --lr 1e-4 \
    --weight-decay 1e-8 \
    --epochs 100 \
    --min-lr 5e-6 \
    --batch-size $BATCH_SIZE \
    --output_dir ./exp/lowrank/$MODEL_SIZE \
    --svd_type 'mix_head_fc_qk'

# STEP 2: Extract attention maps
echo ""
echo "===== STEP 2: Extract Attention Maps ====="
python main.py \
    --eval \
    --model $MODEL \
    --finetune ./exp/lowrank/$MODEL_SIZE/checkpoint_best.pth \
    --data-path $DATA_PATH \
    --svd_type 'mix_head_fc_qk' \
    --need_weight \
    --output_dir ./attn/${MODEL_SIZE}_lowrank \
    --batch-size $BATCH_SIZE

# STEP 3: Generate masks
echo ""
echo "===== STEP 3: Generate Sparse Masks ====="
python gen_mask.py \
    --method 'info' \
    --attn "./attn/${MODEL_SIZE}_lowrank/attention_score_pruned.npy" \
    --info_cut 0.9 \
    --output_dir "./masks/$MODEL_SIZE"

# STEP 4: Sparse finetuning
echo ""
echo "===== STEP 4: Low-rank + Sparse Finetuning ====="
python main.py \
    --model $MODEL \
    --resume ./exp/lowrank/$MODEL_SIZE/checkpoint_best.pth \
    --data-path $DATA_PATH \
    --lr 1e-5 \
    --weight-decay 1e-8 \
    --epochs 100 \
    --min-lr 1e-5 \
    --batch-size $BATCH_SIZE \
    --output_dir ./exp/lowrank_sparse/${MODEL_SIZE}_info90 \
    --mask_path "./masks/$MODEL_SIZE/info_0.9.npy" \
    --svd_type 'mix_head_fc_qk' \
    --restart_finetune

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Final model: ./exp/lowrank_sparse/${MODEL_SIZE}_info90/checkpoint_best.pth"
echo "=========================================="