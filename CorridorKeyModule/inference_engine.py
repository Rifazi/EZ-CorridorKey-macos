import logging
import math
import os
import time
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
from .core.model_transformer import GreenFormer
from .core import color_utils as cu

# Cache for linearized checkerboard backgrounds keyed by (width, height)
_bg_lin_cache: dict[tuple[int, int], np.ndarray] = {}

def _patch_hiera_global_attention(hiera_model: nn.Module) -> int:
    """Monkey-patch MaskUnitAttention.forward on global-attention blocks.

    Hiera's MaskUnitAttention creates Q/K/V with shape
    [B, heads, num_windows, N, head_dim]. When num_windows == 1
    (global attention), this 5-D non-contiguous tensor causes PyTorch's
    SDPA to silently fall back to the VRAM-hungry math backend.

    This patch forces Q/K/V to standard 4-D contiguous tensors, enabling
    FlashAttention and dropping VRAM usage per block dramatically.

    Credit: Jhe Kimchi (Discord contribution)
    """
    patched = 0

    for blk in hiera_model.blocks:
        attn = blk.attn

        # Only patch global attention blocks — windowed attention is fine
        if attn.use_mask_unit_attn:
            continue

        def _make_patched_forward(original_attn):
            def _patched_forward(self, x: torch.Tensor) -> torch.Tensor:
                B, N, _ = x.shape
                qkv = self.qkv(x)
                qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
                q, k, v = qkv.unbind(0)             # each [B, heads, N, head_dim]

                if self.q_stride > 1:
                    q = q.reshape(
                        B, self.heads, self.q_stride, -1, self.head_dim
                    ).amax(dim=2)

                # Force contiguous layout so SDPA can use FlashAttention
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()

                x = F.scaled_dot_product_attention(q, k, v)
                x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
                x = self.proj(x)
                return x

            return types.MethodType(_patched_forward, original_attn)

        attn.forward = _make_patched_forward(attn)
        patched += 1

    return patched

class CorridorKeyEngine:
    # VRAM threshold for optimization profile selection.
    # Below this: tiled refiner + selective compile. Above: full-frame compile.
    _VRAM_TILE_THRESHOLD_GB = 12

    # Optimization modes:
    #   'auto'  — detect VRAM and pick best strategy (default)
    #   'speed' — torch.compile, no tiling (12GB+ VRAM)
    #   'lowvram' — tiled refiner + compiled tile kernel (8GB GPUs)
    VALID_OPT_MODES = ('auto', 'speed', 'lowvram')

    def __init__(self, checkpoint_path, device='cuda', img_size=2048, use_refiner=True,
                 optimization_mode='auto', tile_overlap=128, on_status=None):
        logger.info("CorridorKeyEngine.__init__: begin")
        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner
        self.tile_overlap = tile_overlap
        self._on_status = on_status

        # Allow env var override: CORRIDORKEY_OPT_MODE=speed|lowvram|auto
        env_mode = os.environ.get('CORRIDORKEY_OPT_MODE', '').lower()
        if env_mode in self.VALID_OPT_MODES:
            optimization_mode = env_mode
            logger.info(f"Optimization mode override from env: {env_mode}")

        # Resolve optimization profile.
        # Low-VRAM mode keeps tiling, but graph-breaks the tile scheduler and
        # compiles the fixed-shape tile CNN separately.
        #
        # NOTE: MPS (Apple Silicon) requires tiling even with plenty of memory
        # because it doesn't parallelize well across large tensors.
        # _get_vram_gb() uses pynvml first (CUDA), then torch.cuda, then MPS.
        if optimization_mode == 'speed':
            self.tile_size = 0
            self._use_compile = True
            logger.info("Optimization: speed mode (torch.compile, no tiling)")
        elif optimization_mode == 'lowvram':
            self.tile_size = 512
            self._use_compile = True
            logger.info("Optimization: low-VRAM mode (tiled refiner 512x512 + selective torch.compile)")
        else:  # auto
            logger.info("Optimization: auto mode - probing device...")
            # MPS always needs tiling for good performance
            if self.device.type == 'mps':
                self.tile_size = 512
                self._use_compile = True
                logger.info("Optimization: MPS detected -> low-VRAM mode (forced tiling for MPS)")
            else:
                vram_gb = self._get_vram_gb()
                if vram_gb > 0 and vram_gb < self._VRAM_TILE_THRESHOLD_GB:
                    self.tile_size = 512
                    self._use_compile = True
                    logger.info(f"Optimization: auto -> low-VRAM mode "
                                f"({vram_gb:.1f} GB < {self._VRAM_TILE_THRESHOLD_GB} GB threshold)")
                else:
                    self.tile_size = 0
                    self._use_compile = True
                    logger.info(f"Optimization: auto -> speed mode "
                                f"({vram_gb:.1f} GB >= {self._VRAM_TILE_THRESHOLD_GB} GB threshold)")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        logger.info("CorridorKeyEngine.__init__: entering _load_model")
        self.model = self._load_model()

    @staticmethod
    def _get_vram_gb() -> float:
        """Return total GPU VRAM in GB.

        Tries NVML (CUDA), torch.cuda, then MPS.
        """
        logger.debug("_get_vram_gb: entering")
        try:
            import pynvml
            logger.debug("_get_vram_gb: pynvml imported, calling nvmlInit...")
            pynvml.nvmlInit()
            logger.debug("_get_vram_gb: nvmlInit done, getting handle...")
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            logger.debug("_get_vram_gb: got handle, querying memory...")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram = mem.total / (1024 ** 3)
            logger.debug(f"_get_vram_gb: pynvml reports {vram:.1f} GB")
            return vram
        except Exception as e:
            logger.debug(f"_get_vram_gb: pynvml failed: {e}")

        try:
            logger.debug("_get_vram_gb: falling back to torch.cuda...")
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                logger.debug(f"_get_vram_gb: torch.cuda reports {vram:.1f} GB")
                return vram
        except Exception as e:
            logger.debug(f"_get_vram_gb: torch.cuda failed: {e}")

        # MPS (Apple Silicon) — use psutil to estimate shared unified memory
        try:
            logger.debug("_get_vram_gb: checking for MPS device...")
            if torch.backends.mps.is_available():
                import psutil
                # MPS uses unified memory; return total system memory as approximation
                mem_info = psutil.virtual_memory()
                vram = mem_info.total / (1024 ** 3)
                logger.debug(f"_get_vram_gb: MPS detected, total memory: {vram:.1f} GB")
                return vram
        except Exception as e:
            logger.debug(f"_get_vram_gb: MPS probe failed: {e}")

        logger.debug("_get_vram_gb: all probes failed, returning 0")
        return 0.0
        
    def _status(self, msg: str) -> None:
        """Emit status to UI callback and log."""
        logger.info(msg)
        if self._on_status:
            self._on_status(msg)

    def _load_model(self):
        import time as _time
        import logging as _logging

        def _diag(msg):
            """Force-flush diagnostic — visible in log file immediately."""
            logger.info(msg)
            for h in _logging.getLogger().handlers:
                h.flush()

        _diag("_load_model ENTERED")
        logger.info(f"Loading CorridorKey from {self.checkpoint_path}...")

        # Step 1: Initialize backbone
        _diag("Step 1: GreenFormer init...")
        self._status("Initializing model backbone...")
        t0 = _time.monotonic()
        model = GreenFormer(encoder_name='hiera_base_plus_224.mae_in1k_ft_in1k',
                          img_size=self.img_size,
                          use_refiner=self.use_refiner)
        _diag(f"Step 1 done: {_time.monotonic() - t0:.1f}s")
        logger.info(f"GreenFormer init: {_time.monotonic() - t0:.1f}s")

        # Step 2: Move to GPU
        _diag("Step 2: model.to(device)...")
        self._status("Moving model to GPU...")
        t0 = _time.monotonic()
        model = model.to(self.device)
        model.eval()
        _diag(f"Step 2 done: {_time.monotonic() - t0:.1f}s")
        logger.info(f"Model to device: {_time.monotonic() - t0:.1f}s")

        # Step 3: Load checkpoint
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        _diag("Step 3: torch.load checkpoint...")
        self._status("Loading checkpoint weights...")
        t0 = _time.monotonic()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        _diag(f"Step 3 done: {_time.monotonic() - t0:.1f}s")
        logger.info(f"Checkpoint loaded: {_time.monotonic() - t0:.1f}s")

        # Step 4: Fix Compiled Model Prefix & Handle PosEmbed Mismatch
        new_state_dict = {}
        model_state = model.state_dict()

        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                k = k[10:]

            # Check for PosEmbed Mismatch
            if 'pos_embed' in k and k in model_state:
                if v.shape != model_state[k].shape:
                    logger.warning(f"PosEmbed shape mismatch: resizing {k} from {v.shape} to {model_state[k].shape}")
                    N_src = v.shape[1]
                    N_dst = model_state[k].shape[1]
                    C = v.shape[2]

                    grid_src = int(math.sqrt(N_src))
                    grid_dst = int(math.sqrt(N_dst))

                    v_img = v.permute(0, 2, 1).view(1, C, grid_src, grid_src)
                    v_resized = F.interpolate(v_img, size=(grid_dst, grid_dst), mode='bicubic', align_corners=False)
                    v = v_resized.flatten(2).transpose(1, 2)

            new_state_dict[k] = v

        _diag("Step 4: load_state_dict...")
        self._status("Loading state dict...")
        t0 = _time.monotonic()
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            logger.warning(f"Missing keys in checkpoint: {missing}")
        if len(unexpected) > 0:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
        _diag(f"Step 4 done: {_time.monotonic() - t0:.1f}s")
        logger.info(f"State dict loaded: {_time.monotonic() - t0:.1f}s")

        # Enable TF32 tensor cores for FP32 matmuls (Ampere+, CUDA only).
        if self.device.type != 'mps':
            try:
                torch.set_float32_matmul_precision('high')
                logger.info("TF32 matmul precision set to 'high'")
            except Exception as e:
                logger.debug(f"TF32 precision not available: {e}")

        # Disable cuDNN benchmark to prevent workspace memory allocation (2-5 GB, CUDA only).
        if self.device.type == 'cuda':
            try:
                torch.backends.cudnn.benchmark = False
                logger.info("cuDNN benchmark disabled (saves 2-5 GB workspace)")
            except Exception as e:
                logger.debug(f"cuDNN benchmark disable failed: {e}")

        # Step 5: Hiera attention patch
        self._status("Patching attention blocks...")
        t0 = _time.monotonic()
        try:
            hiera = model.encoder.model
            n_patched = _patch_hiera_global_attention(hiera)
            logger.info(f"Hiera attention patch: {n_patched} blocks ({_time.monotonic() - t0:.1f}s)")
        except Exception as e:
            logger.warning(f"Hiera attention patch failed: {type(e).__name__}: {e}")

        # Configure tiled refiner for VRAM-constrained processing.
        if self.tile_size > 0 and hasattr(model, 'refiner') and model.refiner is not None:
            model.refiner._tile_size = self.tile_size
            model.refiner._tile_overlap = self.tile_overlap
            logger.info(f"Tiled refiner: {self.tile_size}x{self.tile_size} tiles, {self.tile_overlap}px overlap")

        # Step 6: torch.compile (Triton JIT — slow on first run)
        if self._use_compile:
            self._status("Compiling model (first run may take a minute)...")
            import subprocess
            import sys

            if sys.platform == 'win32' and not getattr(subprocess.Popen, '_corridorkey_no_window', False):
                _orig_popen_init = subprocess.Popen.__init__

                def _silent_popen_init(self, *args, **kwargs):
                    kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
                    _orig_popen_init(self, *args, **kwargs)

                subprocess.Popen.__init__ = _silent_popen_init
                subprocess.Popen._corridorkey_no_window = True

            if self.tile_size > 0 and hasattr(model, 'refiner') and model.refiner is not None:
                try:
                    t0 = _time.monotonic()
                    model.refiner.compile_tile_kernel()
                    logger.info(f"Refiner tile compile: {_time.monotonic() - t0:.1f}s")
                except Exception as e:
                    logger.warning(
                        f"Refiner tile kernel compile failed (falling back to eager tiles): "
                        f"{type(e).__name__}: {e}"
                    )

            # Skip torch.compile on MPS (TorchDynamo has issues with MPS)
            device = next(model.parameters()).device
            if device.type == 'mps':
                logger.info("Skipping torch.compile on MPS device (TorchDynamo not well-supported)")
            else:
                try:
                    t0 = _time.monotonic()
                    model = torch.compile(model, fullgraph=False)
                    logger.info(f"torch.compile complete: {_time.monotonic() - t0:.1f}s")
                except Exception as e:
                    logger.warning(f"torch.compile failed (falling back to eager): {type(e).__name__}: {e}")

        self._status("Model ready")
        return model

    def _prepare_input(self, image, mask_linear, input_is_linear=False):
        """Prepare a single image+mask for the model. Returns (H, W, 4) float32."""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0
        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        # Use INTER_AREA for downscaling (faster & better quality), INTER_LINEAR for upscaling
        h, w = image.shape[:2]
        interp = cv2.INTER_AREA if (w > self.img_size or h > self.img_size) else cv2.INTER_LINEAR
        
        if input_is_linear:
            img_resized_lin = cv2.resize(image, (self.img_size, self.img_size), interpolation=interp)
            img_resized = cu.linear_to_srgb(img_resized_lin)
        else:
            img_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=interp)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=interp)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        img_norm = (img_resized - self.mean) / self.std
        return np.concatenate([img_norm, mask_resized], axis=-1)

    @torch.no_grad()
    def predict_raw(self, image, mask_linear, refiner_scale=1.0, input_is_linear=False):
        """Run model inference and return raw alpha/fg at original resolution.

        Args:
            image: Numpy array [H, W, 3] (0.0-1.0 or 0-255).
            mask_linear: Numpy array [H, W] or [H, W, 1] (0.0-1.0).
            refiner_scale: Multiplier for Refiner Deltas.
            input_is_linear: If True, input is linear.

        Returns:
            dict with 'alpha' [H, W, 1] and 'fg' [H, W, 3] float32.
        """
        t0 = time.monotonic()
        h, w = image.shape[:2]
        inp_np = self._prepare_input(image, mask_linear, input_is_linear)
        t_prep = time.monotonic()

        inp_t = torch.from_numpy(inp_np.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        t_to_device = time.monotonic()

        refiner_scale_t = inp_t.new_tensor(refiner_scale)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            out = self.model(inp_t, refiner_scale=refiner_scale_t)

        # Sync MPS before timing — MPS ops are async, .cpu() would block anyway
        if self.device.type == 'mps':
            torch.mps.synchronize()
        t_model = time.monotonic()

        res_alpha = out['alpha'][0].permute(1, 2, 0).float().cpu().numpy()
        res_fg = out['fg'][0].permute(1, 2, 0).float().cpu().numpy()
        t_to_cpu = time.monotonic()

        # Stack and resize together for speed (~8% faster than separate resizes)
        stacked = np.concatenate([res_alpha, res_fg], axis=-1)  # [H, W, 4]
        stacked_resized = cv2.resize(stacked, (w, h), interpolation=cv2.INTER_LANCZOS4)
        res_alpha = stacked_resized[:, :, :1]
        res_fg = stacked_resized[:, :, 1:4]
        t_resize = time.monotonic()

        if res_alpha.ndim == 2:
            res_alpha = res_alpha[:, :, np.newaxis]

        logger.info(
            "predict_raw %dx%d→%d: prep=%.0fms to_gpu=%.0fms model=%.0fms to_cpu=%.0fms resize=%.0fms total=%.0fms",
            w, h, self.img_size,
            (t_prep - t0) * 1000, (t_to_device - t_prep) * 1000,
            (t_model - t_to_device) * 1000, (t_to_cpu - t_model) * 1000,
            (t_resize - t_to_cpu) * 1000, (t_resize - t0) * 1000,
        )

        # Free GPU tensors promptly
        del inp_t, out
        if self.device.type == 'mps':
            torch.mps.empty_cache()

        return {'alpha': res_alpha, 'fg': res_fg}

    @torch.no_grad()
    def predict_raw_batch(self, images, masks, refiner_scale=1.0, input_is_linear=False):
        """Process multiple crops through the model sequentially.

        The Hiera encoder uses .view() internally which requires batch=1.
        This method pre-computes all input tensors on CPU, then runs them
        through the GPU one at a time with minimal Python overhead between
        calls, and batches the resize-back on CPU.

        Args:
            images: list of [H_i, W_i, 3] numpy arrays.
            masks: list of [H_i, W_i] or [H_i, W_i, 1] numpy arrays.
            refiner_scale: Refiner delta multiplier.
            input_is_linear: Whether inputs are linear color.

        Returns:
            list of dicts, each with 'alpha' [H_i, W_i, 1] and 'fg' [H_i, W_i, 3].
        """
        n = len(images)
        if n == 0:
            return []

        # For tile batches, determine optimal canvas size based on actual tile dimensions
        # This makes inference fast for small tiles and good-quality for large tiles
        orig_sizes = [(img.shape[1], img.shape[0]) for img in images]
        max_tile_dim = max(max(h, w) for w, h in orig_sizes) if orig_sizes else 256
        
        # Choose canvas size proportional to max tile size
        if self.device.type == 'mps':
            # MPS: be conservative with memory
            if max_tile_dim <= 128:
                batch_img_size = 384   # 3x upscaling
            elif max_tile_dim <= 256:
                batch_img_size = 512   # 2x upscaling  
            elif max_tile_dim <= 512:
                batch_img_size = 768   # 1.5x upscaling
            else:
                batch_img_size = self.img_size  # Use configured size for large tiles
        else:
            # CUDA: use full res
            batch_img_size = self.img_size
        
        logger.debug(f"predict_raw_batch: max_tile_dim={max_tile_dim}px, using img_size={batch_img_size}px for {n} tiles")
        
        # Temporarily override img_size for this batch
        orig_img_size = self.img_size
        self.img_size = batch_img_size

        # Pre-compute all input tensors on CPU (numpy ops, no GPU wait)
        input_tensors = []
        for img, msk in zip(images, masks):
            inp_np = self._prepare_input(img, msk, input_is_linear)
            inp_t = torch.from_numpy(inp_np.transpose((2, 0, 1))).float().unsqueeze(0)
            input_tensors.append(inp_t)

        # Run each tile through GPU sequentially (model requires batch=1)
        t_batch_start = time.monotonic()
        refiner_scale_val = refiner_scale
        raw_alphas = []
        raw_fgs = []
        for tile_i, inp_t in enumerate(input_tensors):
            t_tile = time.monotonic()
            inp_t = inp_t.to(self.device)
            rs_t = inp_t.new_tensor(refiner_scale_val)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                out = self.model(inp_t, refiner_scale=rs_t)
            if self.device.type == 'mps':
                torch.mps.synchronize()
            raw_alphas.append(out['alpha'][0].float().cpu())
            raw_fgs.append(out['fg'][0].float().cpu())
            del inp_t, out
            logger.debug("  tile %d/%d: %.0fms", tile_i + 1, n, (time.monotonic() - t_tile) * 1000)

        if self.device.type == 'mps':
            torch.mps.empty_cache()

        # Resize all results back on CPU (stack then resize for speed)
        results = []
        for i in range(n):
            w, h = orig_sizes[i]
            alpha_np = raw_alphas[i].permute(1, 2, 0).contiguous().numpy()
            fg_np = raw_fgs[i].permute(1, 2, 0).contiguous().numpy()
            # Stack and resize together (~8% faster)
            stacked = np.concatenate([alpha_np, fg_np], axis=-1)  # [H, W, 4]
            stacked_resized = cv2.resize(stacked, (w, h), interpolation=cv2.INTER_LANCZOS4)
            alpha_np = stacked_resized[:, :, :1]
            fg_np = stacked_resized[:, :, 1:4]
            if alpha_np.ndim == 2:
                alpha_np = alpha_np[:, :, np.newaxis]
            results.append({'alpha': alpha_np, 'fg': fg_np})

        # Restore original img_size
        self.img_size = orig_img_size
        
        logger.info("predict_raw_batch: %d tiles in %.0fms", n, (time.monotonic() - t_batch_start) * 1000)
        return results

    @staticmethod
    def finalize_outputs(res_alpha, res_fg, fg_is_straight=True, despill_strength=1.0,
                         auto_despeckle=True, despeckle_size=400, despeckle_dilation=25,
                         despeckle_blur=5, comp_enabled=True, processed_enabled=True):
        """Post-process stitched raw alpha/fg into final outputs.

        This is CPU-only (numpy/OpenCV) and should be called once on the
        full stitched frame, not per-tile.

        Args:
            res_alpha: [H, W, 1] float32 raw alpha prediction.
            res_fg: [H, W, 3] float32 sRGB raw foreground prediction.
            fg_is_straight: True if FG is straight (unpremultiplied).
            despill_strength: 0.0-1.0 despill multiplier.
            auto_despeckle: Clean up small disconnected alpha islands.
            despeckle_size: Min pixel area to keep.
            despeckle_dilation: Dilation radius for clean_matte.
            despeckle_blur: Blur kernel half-size for clean_matte.
            comp_enabled: If False, skip checkerboard/composite (comp=None).
            processed_enabled: If False, skip premultiply/RGBA concat (processed=None).

        Returns:
            dict with 'alpha', 'fg', 'comp', 'processed'.
        """
        t_total = time.monotonic()
        h, w = res_alpha.shape[:2]

        # --- clean matte ---
        t0 = time.monotonic()
        if auto_despeckle:
            processed_alpha = cu.clean_matte(res_alpha, area_threshold=despeckle_size,
                                             dilation=despeckle_dilation, blur_size=despeckle_blur)
        else:
            processed_alpha = res_alpha
        dt_matte = time.monotonic() - t0

        # --- despill ---
        t0 = time.monotonic()
        if despill_strength > 0.0:
            fg_despilled = cu.despill(res_fg, green_limit_mode='average', strength=despill_strength)
        else:
            fg_despilled = res_fg
        dt_despill = time.monotonic() - t0

        # --- srgb_to_linear (only needed for comp or processed) ---
        dt_color = 0.0
        fg_despilled_lin = None
        if comp_enabled or processed_enabled:
            t0 = time.monotonic()
            fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
            dt_color = time.monotonic() - t0

        # --- premultiply + RGBA concat (only for processed) ---
        dt_premul = 0.0
        processed_rgba = None
        if processed_enabled:
            t0 = time.monotonic()
            fg_premul_lin = cu.premultiply(fg_despilled_lin, processed_alpha)
            processed_rgba = np.concatenate([fg_premul_lin, processed_alpha], axis=-1)
            dt_premul = time.monotonic() - t0

        # --- composite (only for comp) ---
        dt_comp = 0.0
        comp_srgb = None
        if comp_enabled:
            t0 = time.monotonic()
            key = (w, h)
            bg_lin = _bg_lin_cache.get(key)
            if bg_lin is None:
                bg_srgb = cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55)
                bg_lin = cu.srgb_to_linear(bg_srgb)
                _bg_lin_cache[key] = bg_lin

            if fg_is_straight:
                comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
            else:
                comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)

            comp_srgb = cu.linear_to_srgb(comp_lin)
            dt_comp = time.monotonic() - t0

        dt_total = time.monotonic() - t_total
        logger.debug(
            "finalize %dx%d: matte=%.0fms despill=%.0fms color=%.0fms premul=%.0fms comp=%.0fms total=%.0fms",
            w, h,
            dt_matte * 1000, dt_despill * 1000, dt_color * 1000,
            dt_premul * 1000, dt_comp * 1000, dt_total * 1000,
        )

        return {
            'alpha': res_alpha,
            'fg': res_fg,
            'comp': comp_srgb,
            'processed': processed_rgba,
        }

    @torch.no_grad()
    def process_frame(self, image, mask_linear, refiner_scale=1.0, input_is_linear=False, fg_is_straight=True, despill_strength=1.0, auto_despeckle=True, despeckle_size=400, despeckle_dilation=25, despeckle_blur=5, comp_enabled=True, processed_enabled=True):
        """
        Process a single frame (full pipeline: predict + finalize).
        Args:
            image: Numpy array [H, W, 3] (0.0-1.0 or 0-255).
                   - If input_is_linear=False (Default): Assumed sRGB.
                   - If input_is_linear=True: Assumed Linear.
            mask_linear: Numpy array [H, W] or [H, W, 1] (0.0-1.0). Assumed Linear.
            refiner_scale: Multiplier for Refiner Deltas (default 1.0).
            input_is_linear: bool. If True, resizes in Linear then transforms to sRGB.
                             If False, resizes in sRGB (standard).
            fg_is_straight: bool. If True, assumes FG output is Straight (unpremultiplied).
                            If False, assumes FG output is Premultiplied.
            despill_strength: float. 0.0 to 1.0 multiplier for the despill effect.
            auto_despeckle: bool. If True, cleans up small disconnected components from the predicted alpha matte.
            despeckle_size: int. Minimum number of consecutive pixels required to keep an island.
        Returns:
             dict: {'alpha': np, 'fg': np (sRGB), 'comp': np (sRGB on Gray)}
        """
        t0 = time.monotonic()

        raw = self.predict_raw(image, mask_linear, refiner_scale=refiner_scale,
                               input_is_linear=input_is_linear)

        result = self.finalize_outputs(
            raw['alpha'], raw['fg'],
            fg_is_straight=fg_is_straight,
            despill_strength=despill_strength,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
            despeckle_dilation=despeckle_dilation,
            despeckle_blur=despeckle_blur,
            comp_enabled=comp_enabled,
            processed_enabled=processed_enabled,
        )

        h, w = raw['alpha'].shape[:2]
        logger.debug(f"process_frame: {h}x{w} in {time.monotonic() - t0:.3f}s")

        return result
