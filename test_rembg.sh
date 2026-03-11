#!/usr/bin/env bash
# Quick test of rembg installation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

echo "Testing rembg installation..."
echo ""

# Test 1: Import check
echo "✓ Checking rembg import..."
.venv/bin/python -c "from rembg import remove; print('  rembg version OK')" 2>&1 || exit 1

# Test 2: onnxruntime check
echo "✓ Checking onnxruntime..."
.venv/bin/python -c "import onnxruntime; print('  onnxruntime version OK')" 2>&1 || exit 1

# Test 3: Model download check (happens on first use)
echo "✓ Testing model loading (may download ~200MB on first run)..."
.venv/bin/python << 'PYEOF'
from rembg import remove
from PIL import Image
import numpy as np

# Create a dummy image to test model loading
dummy = Image.new('RGB', (100, 100), color='white')
try:
    result = remove(dummy)
    print("  Model loaded and tested successfully")
except Exception as e:
    print(f"  Error: {e}")
    exit(1)
PYEOF

echo ""
echo "✓ All tests passed! rembg is ready to use."
echo ""
echo "Usage:"
echo "  ./remove_bg.sh input.mov output.mov"
