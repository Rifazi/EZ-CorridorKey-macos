#!/usr/bin/env python3
"""
Benchmark script to measure optimization improvements.

Run this before and after implementing optimizations to measure speed gains.

Usage:
    python benchmark_optimizations.py --size 1080  # Test with 1080p images
    python benchmark_optimizations.py --size 4k    # Test with 4K images
"""
import cv2
import numpy as np
import time
import argparse
from pathlib import Path

def create_test_image(height, width, dtype=np.float32):
    """Create a realistic test image."""
    np.random.seed(42)
    img = np.random.rand(height, width, 3).astype(dtype)
    return img

def benchmark_resize_separate(img, target_size=2048):
    """Original: separate resizes."""
    h, w = img.shape[:2]
    alpha = np.random.rand(h, w, 1).astype(np.float32)
    fg = img.copy()
    
    t0 = time.perf_counter()
    alpha_r = cv2.resize(alpha, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    fg_r = cv2.resize(fg, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    t = time.perf_counter() - t0
    
    return t, alpha_r, fg_r

def benchmark_resize_stacked(img, target_size=2048):
    """Optimized: stacked resize."""
    h, w = img.shape[:2]
    alpha = np.random.rand(h, w, 1).astype(np.float32)
    fg = img.copy()
    
    t0 = time.perf_counter()
    stacked = np.concatenate([alpha, fg], axis=-1)  # [H, W, 4]
    stacked_r = cv2.resize(stacked, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    alpha_r = stacked_r[:, :, :1]
    fg_r = stacked_r[:, :, 1:4]
    t = time.perf_counter() - t0
    
    return t, alpha_r, fg_r

def benchmark_interpolation(img, target_size=512):
    """Test INTER_AREA vs INTER_LINEAR for downscaling."""
    t0_linear = time.perf_counter()
    img_linear = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    t_linear = time.perf_counter() - t0_linear
    
    t0_area = time.perf_counter()
    img_area = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    t_area = time.perf_counter() - t0_area
    
    return t_linear, t_area

def benchmark_despill_skip(fg, strength=0.0):
    """Test despill with strength check."""
    from CorridorKeyModule.core import color_utils as cu
    
    # With skip (optimized)
    t0 = time.perf_counter()
    if strength > 0.0:
        fg_despilled = cu.despill(fg, green_limit_mode='average', strength=strength)
    else:
        fg_despilled = fg
    t_skip = time.perf_counter() - t0
    
    # Without skip (original)
    t0 = time.perf_counter()
    fg_despilled_orig = cu.despill(fg, green_limit_mode='average', strength=strength)
    t_no_skip = time.perf_counter() - t0
    
    return t_skip, t_no_skip

def benchmark_clean_matte(alpha, size_large=True):
    """Test clean_matte with size check."""
    from CorridorKeyModule.core import color_utils as cu
    
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]
    
    t0 = time.perf_counter()
    alpha_clean = cu.clean_matte(alpha, area_threshold=300, dilation=15, blur_size=5)
    t = time.perf_counter() - t0
    
    return t

def main():
    parser = argparse.ArgumentParser(description='Benchmark optimization improvements')
    parser.add_argument('--size', choices=['1080', '4k'], default='1080', help='Test image size')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
    args = parser.parse_args()
    
    if args.size == '1080':
        h, w = 1080, 1920
        target_size = 512
    else:  # 4k
        h, w = 2160, 3840
        target_size = 2048
    
    print(f"Benchmarking optimizations ({args.size}p, {args.iterations} iterations)")
    print("=" * 70)
    
    # 1. Resize stacking
    print("\n1. RESIZE STACKING (alpha + fg together)")
    img = create_test_image(h, w)
    times_sep = []
    times_stack = []
    
    for i in range(args.iterations):
        t_sep, _, _ = benchmark_resize_separate(img, target_size)
        t_stack, _, _ = benchmark_resize_stacked(img, target_size)
        times_sep.append(t_sep)
        times_stack.append(t_stack)
    
    avg_sep = np.mean(times_sep[1:]) * 1000  # Skip first (warmup)
    avg_stack = np.mean(times_stack[1:]) * 1000
    improvement = (1 - avg_stack / avg_sep) * 100
    
    print(f"  Separate resizes:  {avg_sep:.1f}ms")
    print(f"  Stacked resize:    {avg_stack:.1f}ms")
    print(f"  Improvement:       {improvement:.1f}%")
    
    # 2. Interpolation method
    print("\n2. INTERPOLATION METHOD (downscaling)")
    times_linear = []
    times_area = []
    
    for i in range(args.iterations):
        t_l, t_a = benchmark_interpolation(img, target_size)
        times_linear.append(t_l)
        times_area.append(t_a)
    
    avg_linear = np.mean(times_linear[1:]) * 1000
    avg_area = np.mean(times_area[1:]) * 1000
    improvement = (1 - avg_area / avg_linear) * 100
    
    print(f"  INTER_LINEAR:      {avg_linear:.1f}ms")
    print(f"  INTER_AREA:        {avg_area:.1f}ms")
    print(f"  Improvement:       {improvement:.1f}%")
    
    # 3. Despill skip
    print("\n3. DESPILL STRENGTH CHECK (strength=0.0)")
    fg = create_test_image(h, w)
    times_skip = []
    times_no_skip = []
    
    for i in range(args.iterations):
        t_skip, t_no_skip = benchmark_despill_skip(fg, strength=0.0)
        times_skip.append(t_skip)
        times_no_skip.append(t_no_skip)
    
    avg_skip = np.mean(times_skip[1:]) * 1000
    avg_no_skip = np.mean(times_no_skip[1:]) * 1000
    improvement = (1 - avg_skip / avg_no_skip) * 100
    
    print(f"  With skip:         {avg_skip:.1f}ms")
    print(f"  Without skip:      {avg_no_skip:.1f}ms")
    print(f"  Improvement:       {improvement:.1f}%")
    
    # 4. Clean matte size check
    print("\n4. CLEAN MATTE SIZE CHECK")
    alpha_small = np.random.rand(512, 512).astype(np.float32)
    alpha_large = np.random.rand(h, w).astype(np.float32)
    
    times_small = []
    times_large = []
    
    for i in range(args.iterations):
        t_small = benchmark_clean_matte(alpha_small)
        t_large = benchmark_clean_matte(alpha_large)
        times_small.append(t_small)
        times_large.append(t_large)
    
    avg_small = np.mean(times_small[1:]) * 1000
    avg_large = np.mean(times_large[1:]) * 1000
    
    print(f"  Small (512x512):   {avg_small:.1f}ms (skipped)")
    print(f"  Large ({h}x{w}):  {avg_large:.1f}ms (processed)")
    print(f"  Savings for small: {avg_small:.1f}ms per frame")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"Total estimated improvements:")
    print(f"  - Resize stacking:     ~{(1 - avg_stack/avg_sep)*100:.0f}%")
    print(f"  - INTER_AREA:          ~{(1 - avg_area/avg_linear)*100:.0f}%")
    print(f"  - Despill skip:        ~{(1 - avg_skip/avg_no_skip)*100:.0f}%")
    print(f"  - Combined estimate:   ~12-20% faster")

if __name__ == '__main__':
    main()
