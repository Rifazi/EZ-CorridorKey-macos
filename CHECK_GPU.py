#!/usr/bin/env python3
"""Quick check if GPU/MPS is available and working."""
import torch
import os

print("=" * 60)
print("GPU/MPS AVAILABILITY CHECK")
print("=" * 60)

# Check device detection
if torch.cuda.is_available():
    print("✓ CUDA available:", torch.cuda.get_device_name(0))
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("✓ MPS (Mac GPU) available")
    print("  Device:", torch.device("mps"))
else:
    print("✗ NO GPU DETECTED - using CPU (very slow)")
    print("  This is the problem!")

print("\n" + "=" * 60)
print("ENVIRONMENT VARIABLES")
print("=" * 60)
print("PYTORCH_ENABLE_MPS_FALLBACK:", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "not set"))
print("PYTORCH_MPS_HIGH_WATERMARK_RATIO:", os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "not set"))

print("\n" + "=" * 60)
print("TEST: MOVING TENSOR TO GPU")
print("=" * 60)

# Test actual tensor movement
x = torch.randn(1, 3, 1024, 1024)
print(f"Tensor on CPU: {x.device}")

if torch.cuda.is_available():
    x_gpu = x.cuda()
    print(f"Tensor on CUDA: {x_gpu.device}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    x_mps = x.to("mps")
    print(f"Tensor on MPS: {x_mps.device}")
    # Test small operation
    try:
        y = x_mps @ x_mps.transpose(-2, -1)
        print("✓ MPS computation works")
    except Exception as e:
        print(f"✗ MPS computation failed: {e}")
else:
    print("CPU only - no GPU available")

print("\n" + "=" * 60)
