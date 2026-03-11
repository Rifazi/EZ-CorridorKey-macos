#!/usr/bin/env python3
"""
Quick profiling script to identify inference bottlenecks.

Usage:
    python3 profile_inference.py <video_path> [--log-level DEBUG]

Example:
    python3 profile_inference.py clips/test.mp4 --log-level INFO
"""

import os
import sys
import logging
import time
import argparse

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(
        description="Profile inference bottlenecks for a video clip"
    )
    parser.add_argument("video_path", help="Path to video file to process")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=24,
        help="Frames per inference chunk (default: 24)"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Process only first N chunks (default: all)"
    )
    parser.add_argument(
        "--mask-mode",
        choices=["vae", "interpolate"],
        default="vae",
        help="Mask conditioning mode (default: vae)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format=fmt,
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("profile_inference")
    
    # Verify video exists
    if not os.path.isfile(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    logger.info(f"Profiling: {args.video_path}")
    logger.info(f"Config: chunk_size={args.chunk_size}, mask_mode={args.mask_mode}")
    
    try:
        from VideoMaMaInferenceModule.inference import (
            extract_frames_from_video,
            load_videomama_model,
            run_inference
        )
        from CorridorKeyModule.inference_engine import CorridorKeyEngine
        
        # Step 1: Extract frames
        logger.info("=" * 60)
        logger.info("STEP 1: Extract frames")
        logger.info("=" * 60)
        t0 = time.monotonic()
        frames, fps = extract_frames_from_video(args.video_path)
        dt_extract = time.monotonic() - t0
        logger.info(f"Extracted {len(frames)} frames @ {fps} fps in {dt_extract:.2f}s\n")
        
        # Limit to N chunks if specified
        if args.num_chunks:
            max_frames = args.num_chunks * args.chunk_size
            frames = frames[:max_frames]
            logger.info(f"Limited to {len(frames)} frames ({args.num_chunks} chunks)\n")
        
        # Step 2: Load VideoMaMa model
        logger.info("=" * 60)
        logger.info("STEP 2: Load VideoMaMa model")
        logger.info("=" * 60)
        t0 = time.monotonic()
        pipeline = load_videomama_model(device="cuda")
        dt_load_model = time.monotonic() - t0
        logger.info(f"Model loaded in {dt_load_model:.2f}s\n")
        
        # Step 3: Generate dummy mask frames (for testing)
        logger.info("=" * 60)
        logger.info("STEP 3: Create dummy mask frames")
        logger.info("=" * 60)
        import numpy as np
        mask_frames = [np.ones((frames[0].shape[0], frames[0].shape[1]), dtype=np.uint8) * 200
                       for _ in frames]
        logger.info(f"Created {len(mask_frames)} mask frames\n")
        
        # Step 4: Run inference with profiling
        logger.info("=" * 60)
        logger.info(f"STEP 4: Run inference ({args.mask_mode} mode)")
        logger.info("=" * 60)
        
        t0 = time.monotonic()
        total_frames = 0
        for i, chunk_output in enumerate(run_inference(
            pipeline,
            frames,
            mask_frames,
            chunk_size=args.chunk_size,
            on_status=lambda msg: logger.info(f"  Status: {msg}")
        ), 1):
            total_frames += len(chunk_output)
            logger.info(f"Chunk {i} complete: {total_frames}/{len(frames)} frames processed\n")
        
        dt_inference = time.monotonic() - t0
        logger.info("=" * 60)
        logger.info("PROFILE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Frame extraction: {dt_extract:.2f}s")
        logger.info(f"Model loading:    {dt_load_model:.2f}s")
        logger.info(f"Inference:        {dt_inference:.2f}s")
        logger.info(f"Total:            {dt_extract + dt_load_model + dt_inference:.2f}s")
        logger.info(f"Throughput:       {len(frames) / dt_inference:.1f} fps")
        logger.info("")
        logger.info(f"Per-frame average: {dt_inference / len(frames) * 1000:.0f}ms")
        logger.info(f"Per-chunk average: {dt_inference / ((len(frames) + args.chunk_size - 1) // args.chunk_size):.2f}s")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during profiling: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
