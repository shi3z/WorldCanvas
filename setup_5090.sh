#!/bin/bash
# Setup script for RTX 5090 inference

set -e

echo "=== WorldCanvas 5090 Setup ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Install Python dependencies
echo "Installing Python dependencies..."
UV_HTTP_TIMEOUT=600 uv sync

# Create checkpoints directory structure
mkdir -p checkpoints/Wan2.2-T2V-A14B
mkdir -p checkpoints/WorldCanvas_dit/WorldCanvas

# Check for existing TurboDiffusion models to symlink
if [ -f ~/TurboDiffusion/checkpoints/Wan2.1_VAE.pth ]; then
    echo "Linking VAE from TurboDiffusion..."
    ln -sf ~/TurboDiffusion/checkpoints/Wan2.1_VAE.pth checkpoints/Wan2.2-T2V-A14B/
fi

if [ -f ~/TurboDiffusion/checkpoints/models_t5_umt5-xxl-enc-bf16.pth ]; then
    echo "Linking text encoder from TurboDiffusion..."
    ln -sf ~/TurboDiffusion/checkpoints/models_t5_umt5-xxl-enc-bf16.pth checkpoints/Wan2.2-T2V-A14B/
fi

# Download SAM model if not present
if [ ! -f checkpoints/sam_vit_h_4b8939.pth ]; then
    echo "Downloading SAM model..."
    wget -q --show-progress -O checkpoints/sam_vit_h_4b8939.pth \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
fi

# Check for WorldCanvas DiT models
if [ ! -f checkpoints/WorldCanvas_dit/WorldCanvas/high_model.safetensors ]; then
    echo ""
    echo "WARNING: WorldCanvas DiT models not found!"
    echo "You need to copy high_model.safetensors and low_model.safetensors to:"
    echo "  checkpoints/WorldCanvas_dit/WorldCanvas/"
    echo ""
    echo "From another machine with models:"
    echo "  rsync -avz --progress /path/to/WorldCanvas/checkpoints/WorldCanvas_dit \\"
    echo "    \$USER@\$(hostname):~/WorldCanvas/checkpoints/"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run WebUI:"
echo "  uv run python gradio/draw_traj_with_gen.py"
echo ""
echo "Access at: http://localhost:8050"
