#!/bin/bash
# WorldCanvas WebUI Launcher
# Access from VPN: http://<server-ip>:7860

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
HOST="0.0.0.0"
PORT=7860
SHARE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --share)
            SHARE="--share"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  WorldCanvas WebUI"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv is not installed. Install with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check for checkpoints
if [ ! -d "checkpoints" ]; then
    echo "[WARNING] checkpoints directory not found!"
    echo "Please download the required models:"
    echo "  - SAM: checkpoints/sam_vit_h_4b8939.pth"
    echo "  - T5: checkpoints/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth"
    echo "  - VAE: checkpoints/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
    echo "  - DiT: checkpoints/WorldCanvas_dit/WorldCanvas/*.safetensors"
    echo ""
fi

# Sync dependencies
echo "[INFO] Syncing dependencies with uv..."
uv sync

# Get local IP for VPN access
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

echo ""
echo "[INFO] Starting WebUI..."
echo "[INFO] Local access: http://localhost:$PORT"
echo "[INFO] VPN access: http://$LOCAL_IP:$PORT"
echo ""

# Run the WebUI
uv run python webui.py --host "$HOST" --port "$PORT" $SHARE
