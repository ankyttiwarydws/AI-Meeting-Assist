#!/usr/bin/env python3
"""Check CUDA and PyTorch setup"""
import torch
import sys

print("=== CUDA/PyTorch Diagnostic ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available - will use CPU")

# Test simple tensor operation
try:
    if torch.cuda.is_available():
        x = torch.randn(3, 3).cuda()
        print("✓ CUDA tensor test passed")
    else:
        x = torch.randn(3, 3)
        print("✓ CPU tensor test passed")
except Exception as e:
    print(f"❌ Tensor test failed: {e}")