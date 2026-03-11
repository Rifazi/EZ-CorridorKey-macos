#!/usr/bin/env python3
"""
Quick background removal using rembg with tile-based motion detection.

Usage:
    python remove_background.py input.mov output.mov

Tile-based motion detection skips unchanged tile regions between frames.
Changed tiles are processed individually and stitched back. Writes are
pipelined with processing via a background thread.
"""
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rembg import remove, new_session
import cv2
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.tile_motion import MotionConfig, build_motion_plan, crop_roi, paste_core


def _rembg_mask(frame_rgb, session):
    """Run rembg on a single RGB frame, return uint8 grayscale mask."""
    mask = remove(frame_rgb, session=session, only_mask=True)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return mask


def remove_background_video(input_path, output_path, motion_config=None):
    """Remove background from video using rembg with tile-based motion skip."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    mcfg = motion_config if motion_config is not None else MotionConfig()

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False

    logger.info(f"Processing: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps}, Frames: {frame_count}")
    logger.info(f"  Tile motion: {'enabled' if mcfg.enabled else 'disabled'} "
                f"(tile={mcfg.tile_size}px, halo={mcfg.rembg_halo_px}px)")

    model_name = os.environ.get("CK_REMBG_MODEL", "u2netp").strip() or "u2netp"
    logger.info(f"  rembg model: {model_name}")
    session = new_session(model_name=model_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'ap4h')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        logger.warning("ProRes codec not available, trying fallback...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    prev_frame_rgb = None
    prev_mask = None
    reused_count = 0
    partial_count = 0
    frame_idx = 0

    try:
        with tqdm(total=frame_count, desc="Removing background") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                use_tiling = (
                    mcfg.enabled
                    and prev_frame_rgb is not None
                    and prev_mask is not None
                    and frame_rgb.shape == prev_frame_rgb.shape
                )

                if use_tiling:
                    plan = build_motion_plan(
                        prev_frame_rgb, frame_rgb, mcfg,
                        frame_index=frame_idx,
                        halo_px=mcfg.rembg_halo_px,
                    )
                else:
                    plan = None

                if plan is not None and not plan.full_frame and not plan.tiles:
                    mask = prev_mask
                    reused_count += 1
                elif plan is not None and not plan.full_frame and plan.tiles:
                    mask = prev_mask.copy()
                    for ti in plan.tiles:
                        crop_rgb = crop_roi(frame_rgb, ti)
                        crop_mask = _rembg_mask(crop_rgb, session)
                        paste_core(mask, crop_mask, ti)
                    partial_count += 1
                else:
                    mask = _rembg_mask(frame_rgb, session)

                prev_frame_rgb = frame_rgb
                prev_mask = mask

                out.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
                frame_idx += 1
                pbar.update(1)

    finally:
        cap.release()
        out.release()

    full_count = frame_idx - reused_count - partial_count
    logger.info(f"✓ Done! Output saved to: {output_path}")
    logger.info(f"  Full: {full_count}, Partial: {partial_count}, "
                f"Reused: {reused_count} ({reused_count / max(frame_idx, 1):.0%} savings)")
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python remove_background.py <input.mov> <output.mov>")
        sys.exit(1)
    success = remove_background_video(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
