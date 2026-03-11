#!/usr/bin/env python3
"""Test green screen detection and skip logic."""

import numpy as np
from backend.tile_motion import MotionConfig, build_motion_plan, _detect_green_screen

# Create a simple test frame with pure green regions
def test_green_detection():
    h, w = 256, 256
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Top-left: pure green
    frame[0:128, 0:128] = [0, 255, 0]
    
    # Top-right: slightly off-green (should not match)
    frame[0:128, 128:256] = [50, 255, 50]
    
    # Bottom-left: off-green red tint (should not match)
    frame[128:256, 0:128] = [100, 200, 100]
    
    # Bottom-right: pure green
    frame[128:256, 128:256] = [0, 255, 0]
    
    # Test detection (using default thresholds from MotionConfig)
    green_mask = _detect_green_screen(frame, green_threshold=80, green_diff_threshold=100)
    
    green_count = int(np.count_nonzero(green_mask))
    total_pixels = h * w
    print(f"Detected green pixels: {green_count} / {total_pixels} ({100*green_count/total_pixels:.1f}%)")
    
    # Should detect at least 25% (two quadrants of pure green, rest has mixed tones)
    assert green_count > total_pixels * 0.2, "Failed to detect pure green regions"
    print("✓ Green screen detection works!")

def test_motion_plan_with_green_skip():
    h, w = 256, 256
    
    # Create prev frame (pure green)
    prev_frame = np.zeros((h, w, 3), dtype=np.uint8)
    prev_frame[:, :] = [0, 255, 0]
    
    # Create curr frame (green + motion in person area)
    curr_frame = prev_frame.copy()
    curr_frame[50:100, 50:100] = [100, 150, 100]  # Person area with motion
    
    config = MotionConfig(skip_green_screen=True)
    plan = build_motion_plan(prev_frame, curr_frame, config, frame_index=0)
    
    print(f"\nMotion plan (skip_green_screen=True):")
    print(f"  Full frame: {plan.full_frame}")
    print(f"  Tiles: {len(plan.tiles)}")
    print(f"  Changed area ratio: {plan.changed_area_ratio:.4f}")
    
    # With green skip, should only detect motion in the person area, not the green background
    assert plan.changed_area_ratio < 0.2, "Should skip large green areas"
    print("✓ Green screen skip in motion plan works!")

def test_without_green_skip():
    h, w = 256, 256
    
    # Create prev frame (pure green)
    prev_frame = np.zeros((h, w, 3), dtype=np.uint8)
    prev_frame[:, :] = [0, 255, 0]
    
    # Create curr frame (slightly different green)
    curr_frame = prev_frame.copy()
    curr_frame[:, :] = [5, 250, 5]  # Slight color variation in green
    
    config = MotionConfig(skip_green_screen=False)
    plan = build_motion_plan(prev_frame, curr_frame, config, frame_index=0)
    
    print(f"\nMotion plan (skip_green_screen=False):")
    print(f"  Full frame: {plan.full_frame}")
    print(f"  Tiles: {len(plan.tiles)}")
    print(f"  Changed area ratio: {plan.changed_area_ratio:.4f}")
    
    print("✓ Comparison test works!")

if __name__ == '__main__':
    test_green_detection()
    test_motion_plan_with_green_skip()
    test_without_green_skip()
    print("\n✓ All tests passed!")
