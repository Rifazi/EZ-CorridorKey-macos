"""Adaptive quadtree motion detection for frame-to-frame delta processing.

Uses a quadtree subdivision: starts with the full frame, recursively splits
regions that contain motion into quadrants, stops splitting when a region
is unchanged or has reached the minimum tile size.  The result is a set of
variable-sized tiles — large tiles where nothing moved, small tiles that
tightly cover only the areas that changed.

Example for a 1024×1024 frame where only a hand moved in one corner:
  - Three 512×512 quadrants: no motion → skipped entirely
  - One 512×512 quadrant: has motion → subdivided into four 256×256
    - Three of those: no motion → skipped
    - One 256×256: has motion → subdivided into four 128×128
      - Two 128×128: no motion → skipped
      - Two 128×128: motion → sent to GPU
  Total: 2 GPU calls instead of 1 full-frame call (saving ~97% of pixels)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionConfig:
    """Tunable parameters for adaptive motion detection."""
    enabled: bool = True
    halo_px: int = 64             # context padding around each tile for GPU
    rembg_halo_px: int = 32       # context padding for rembg ROIs
    diff_threshold: int = 10      # per-pixel channel diff threshold (0-255)
    min_changed_ratio: float = 0.01   # fraction of region pixels that must differ
    max_changed_area_ratio: float = 0.5  # above this, full-frame (no tiling)
    max_gpu_tiles: int = 6        # hard cap on GPU calls per frame
    min_tile_size: int = 128      # smallest tile the quadtree will produce
    periodic_full_refresh: int = 60   # force full-frame every N frames
    skip_green_screen: bool = True   # skip processing pure green regions
    green_threshold: int = 80      # min G value to detect green screen
    green_diff_threshold: int = 100 # max(|R-G|, |B-G|) to be considered pure green
    # Legacy fields kept for compatibility
    tile_size: int = 256
    dynamic_tile_size: bool = True
    dilate_tiles: int = 1
    max_changed_tiles_ratio: float = 0.5


@dataclass
class TileInfo:
    """A single tile region to process on the GPU.

    Coordinates are absolute pixel positions in the original frame.
    The "padded" region (x0,y0)-(x1,y1) includes halo context for the model.
    The "core" region is what gets written back to the output.
    """
    row: int; col: int            # quadtree position (for logging only)
    x0: int; y0: int; x1: int; y1: int      # padded region
    core_x0: int; core_y0: int; core_x1: int; core_y1: int  # write-back region
    tile_size: int = 0            # size of this particular tile


@dataclass
class MotionPlan:
    """Result of motion analysis for a single frame pair."""
    full_frame: bool
    tiles: list[TileInfo]         # variable-sized tiles for GPU inference
    changed_area_ratio: float
    scene_cut: bool = False
    curr_preprocessed: Optional[np.ndarray] = None
    # Keep for compatibility with service.py logging
    changed_tiles: Optional[np.ndarray] = None
    tile_size_used: int = 0       # 0 = mixed sizes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_motion_plan(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    config: MotionConfig,
    frame_index: int = 0,
    halo_px: Optional[int] = None,
    prev_mask: Optional[np.ndarray] = None,
    curr_mask: Optional[np.ndarray] = None,
    prev_blur_cached: Optional[np.ndarray] = None,
) -> MotionPlan:
    """Build an adaptive quadtree motion plan.

    1. Compute full-resolution color-aware diff (once).
    2. Check for early exits (no change, scene cut, periodic refresh).
    3. Recursively subdivide the frame — only regions with motion get
       split into smaller tiles.  Unchanged regions are skipped entirely.
    4. If too many tiles result, fall back to full-frame.
    """
    h, w = curr_frame.shape[:2]
    halo = halo_px if halo_px is not None else config.halo_px

    # --- Compute diff ---
    curr_u8 = _to_uint8(curr_frame)
    curr_blur = cv2.GaussianBlur(curr_u8, (5, 5), 0)

    if prev_blur_cached is not None:
        prev_blur = prev_blur_cached
    else:
        prev_u8 = _to_uint8(prev_frame)
        prev_blur = cv2.GaussianBlur(prev_u8, (5, 5), 0)

    if prev_blur.ndim == 3 and prev_blur.shape[2] >= 3:
        diff = np.max(
            np.abs(prev_blur.astype(np.int16) - curr_blur.astype(np.int16)),
            axis=2,
        ).astype(np.uint8)
    else:
        diff = cv2.absdiff(prev_blur, curr_blur)

    # Also include mask delta if provided
    if prev_mask is not None and curr_mask is not None:
        mask_diff = _mask_diff(prev_mask, curr_mask)
        diff = np.maximum(diff, mask_diff)

    # Binary diff map
    diff_binary = diff > config.diff_threshold
    
    # Detect green screen and exclude from processing
    if config.skip_green_screen:
        green_mask = _detect_green_screen(curr_frame, config.green_threshold, config.green_diff_threshold)
        # Don't process pure green regions even if they have motion
        diff_binary = diff_binary & ~green_mask
        logger.debug(f"Green screen mask: {int(np.count_nonzero(green_mask))} pixels masked")
    
    total_above = int(np.count_nonzero(diff_binary))
    changed_area_ratio = total_above / max(h * w, 1)

    # --- Early exits ---
    _empty = MotionPlan(
        full_frame=False, tiles=[], changed_area_ratio=0.0,
        curr_preprocessed=curr_blur,
    )
    _full = lambda ratio, sc=False: MotionPlan(
        full_frame=True, tiles=[], changed_area_ratio=ratio,
        scene_cut=sc, curr_preprocessed=curr_blur,
    )

    if frame_index > 0 and frame_index % config.periodic_full_refresh == 0:
        return _full(1.0)

    scene_cut = changed_area_ratio > 0.6
    if changed_area_ratio > config.max_changed_area_ratio or scene_cut:
        return _full(changed_area_ratio, scene_cut)

    if total_above == 0:
        return _empty

    # --- Quadtree subdivision ---
    tiles: list[TileInfo] = []
    _quadtree_subdivide(
        diff_binary, 0, 0, w, h,
        config.min_tile_size, config.min_changed_ratio,
        config.max_gpu_tiles, halo,
        w, h, tiles, depth=0,
    )

    if not tiles:
        return _empty

    # If we'd need too many GPU calls, just do full frame
    if len(tiles) > config.max_gpu_tiles:
        logger.info(
            "Quadtree produced %d tiles > max %d → full-frame (%.1f%% area changed)",
            len(tiles), config.max_gpu_tiles, changed_area_ratio * 100,
        )
        return _full(changed_area_ratio)

    # Log the plan
    total_tile_area = sum((t.core_x1 - t.core_x0) * (t.core_y1 - t.core_y0) for t in tiles)
    savings = 1.0 - total_tile_area / (h * w)
    sizes = sorted(set(t.tile_size for t in tiles), reverse=True)
    logger.info(
        "Quadtree plan: %d tiles (%s px), %.0f%% area saved, %.1f%% changed",
        len(tiles),
        "+".join(str(s) for s in sizes),
        savings * 100,
        changed_area_ratio * 100,
    )

    return MotionPlan(
        full_frame=False, tiles=tiles,
        changed_area_ratio=changed_area_ratio,
        curr_preprocessed=curr_blur,
    )


# ---------------------------------------------------------------------------
# Quadtree core
# ---------------------------------------------------------------------------

def _quadtree_subdivide(
    diff_binary: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
    min_size: int,
    min_changed_ratio: float,
    max_tiles: int,
    halo: int,
    frame_w: int, frame_h: int,
    out_tiles: list[TileInfo],
    depth: int = 0,
) -> None:
    """Recursively subdivide a region.  Only regions with motion are split.

    A region is "changed" if the fraction of diff_binary pixels above
    threshold exceeds min_changed_ratio.

    If a region has motion and is larger than min_size, split into 4
    quadrants and recurse.  If it's at min_size (or budget exhausted),
    add it as a tile.
    """
    rw = x1 - x0
    rh = y1 - y0
    if rw <= 0 or rh <= 0:
        return

    # Check if this region has motion
    region = diff_binary[y0:y1, x0:x1]
    changed_count = int(np.count_nonzero(region))
    ratio = changed_count / max(region.size, 1)

    if ratio < min_changed_ratio:
        return  # No meaningful motion — skip this region entirely

    # Budget check: if we're already at the limit, don't add more
    if len(out_tiles) >= max_tiles:
        return

    # Can we subdivide further?
    half_w = rw // 2
    half_h = rh // 2
    can_split = (half_w >= min_size and half_h >= min_size)

    if can_split:
        # Split into 4 quadrants
        mid_x = x0 + half_w
        mid_y = y0 + half_h
        _quadtree_subdivide(diff_binary, x0, y0, mid_x, mid_y,
                            min_size, min_changed_ratio, max_tiles,
                            halo, frame_w, frame_h, out_tiles, depth + 1)
        _quadtree_subdivide(diff_binary, mid_x, y0, x1, mid_y,
                            min_size, min_changed_ratio, max_tiles,
                            halo, frame_w, frame_h, out_tiles, depth + 1)
        _quadtree_subdivide(diff_binary, x0, mid_y, mid_x, y1,
                            min_size, min_changed_ratio, max_tiles,
                            halo, frame_w, frame_h, out_tiles, depth + 1)
        _quadtree_subdivide(diff_binary, mid_x, mid_y, x1, y1,
                            min_size, min_changed_ratio, max_tiles,
                            halo, frame_w, frame_h, out_tiles, depth + 1)
    else:
        # Leaf node — this region needs GPU processing
        # Add halo padding (clamped to frame bounds)
        padded_x0 = max(0, x0 - halo)
        padded_y0 = max(0, y0 - halo)
        padded_x1 = min(frame_w, x1 + halo)
        padded_y1 = min(frame_h, y1 + halo)

        out_tiles.append(TileInfo(
            row=depth, col=len(out_tiles),
            x0=padded_x0, y0=padded_y0,
            x1=padded_x1, y1=padded_y1,
            core_x0=x0, core_y0=y0,
            core_x1=x1, core_y1=y1,
            tile_size=max(rw, rh),
        ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_green_screen(frame: np.ndarray, green_threshold: int, green_diff_threshold: int) -> np.ndarray:
    """Detect pure green screen pixels.
    
    Returns a binary mask where True = green screen, False = non-green.
    A pixel is green screen if:
      - G value >= green_threshold
      - |R - G| <= green_diff_threshold AND |B - G| <= green_diff_threshold
    """
    if frame.ndim != 3 or frame.shape[2] < 3:
        return np.zeros(frame.shape[:2], dtype=bool)
    
    frame_u8 = _to_uint8(frame)
    r = frame_u8[:, :, 0].astype(np.int16)
    g = frame_u8[:, :, 1].astype(np.int16)
    b = frame_u8[:, :, 2].astype(np.int16)
    
    is_green = (
        (g >= green_threshold) &
        (np.abs(r - g) <= green_diff_threshold) &
        (np.abs(b - g) <= green_diff_threshold)
    )
    return is_green


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert any frame to uint8, preserving all channels for color-aware diff."""
    if frame.dtype == np.uint8:
        return frame
    if frame.dtype in (np.float32, np.float64):
        return (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return frame.astype(np.uint8)


def _mask_diff(prev_mask: np.ndarray, curr_mask: np.ndarray) -> np.ndarray:
    """Compute absolute diff between two masks, returned as uint8."""
    p = prev_mask if prev_mask.ndim == 2 else prev_mask[:, :, 0]
    c = curr_mask if curr_mask.ndim == 2 else curr_mask[:, :, 0]
    if p.dtype != np.uint8:
        p = (np.clip(p, 0.0, 1.0) * 255).astype(np.uint8)
    if c.dtype != np.uint8:
        c = (np.clip(c, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.absdiff(p, c)


def crop_roi(frame: np.ndarray, roi) -> np.ndarray:
    """Crop a frame to the padded ROI/TileInfo region."""
    return frame[roi.y0:roi.y1, roi.x0:roi.x1].copy()


def paste_core(target: np.ndarray, patch: np.ndarray, roi) -> None:
    """Paste the core (non-halo) portion of a processed patch back."""
    off_x = roi.core_x0 - roi.x0
    off_y = roi.core_y0 - roi.y0
    core_w = roi.core_x1 - roi.core_x0
    core_h = roi.core_y1 - roi.core_y0
    core_patch = patch[off_y:off_y + core_h, off_x:off_x + core_w]
    target[roi.core_y0:roi.core_y1, roi.core_x0:roi.core_x1] = core_patch
