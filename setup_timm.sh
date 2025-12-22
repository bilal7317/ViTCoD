#!/bin/bash
# setup_timm.sh - Setup timm library with ViTCoD modifications

set -e

cd /home/bilal-linux/Research/ViTCoD
source vitcod_env/bin/activate

# Get timm path
TIMM_PATH=$(python -c "import timm; print(timm.__path__[0])")
echo "Timm installed at: $TIMM_PATH"

# Check if ViTCoD timm files exist
if [ ! -f "Algorithm/deit/timm/vision_transformer.py" ]; then
    echo "Error: ViTCoD timm files not found!"
    exit 1
fi

# Backup original files
echo "Backing up original timm files..."
cp "$TIMM_PATH/models/vision_transformer.py" "$TIMM_PATH/models/vision_transformer.py.backup" 2>/dev/null || true

# Copy ViTCoD files
echo "Copying ViTCoD timm files..."
cp Algorithm/deit/timm/vision_transformer.py "$TIMM_PATH/models/vision_transformer.py"
cp Algorithm/deit/timm/mask_utils.py "$TIMM_PATH/models/mask_utils.py" 2>/dev/null || echo "mask_utils.py not found, skipping"
cp Algorithm/deit/timm/utils.py "$TIMM_PATH/models/timm_utils.py" 2>/dev/null || echo "utils.py not found, skipping"

echo "Done! Timm library updated with ViTCoD modifications."