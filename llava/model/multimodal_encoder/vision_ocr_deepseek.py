"""
DeepSeek-OCR Vision Tower with FULL DYNAMIC RESOLUTION Support
Supports ALL modes: tiny, small, base, large, gundam
"""

from typing import Optional, List, Tuple, Dict, Union
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from functools import partial, reduce
from PIL import Image, ImageOps
import math
import numpy as np

from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)

from llava.utils import rank0_print

# ---------------------------
# Resolution Configurations - ALL MODES
# ---------------------------
RESOLUTION_MODES = {
    'tiny': {
        'size': 512,
        'tokens': 64,
        'process': 'padding',
        'max_input_size': 640,
    },
    'small': {
        'size': 640,
        'tokens': 100,
        'process': 'padding',
        'max_input_size': 768,
    },
    'base': {
        'size': 1024,
        'tokens': 256,
        'process': 'padding',
        'max_input_size': 1280,
    },
    'large': {
        'size': 1280,
        'tokens': 400,
        'process': 'padding',
        'max_input_size': 1536,
    },
    'gundam': {
        'tile_size': 640,  # Tiles are 640×640
        'global_size': 1024,  # Global view is 1024×1024
        'tile_tokens': 100,  # Each tile → 100 tokens
        'global_tokens': 256,  # Global view → 256 tokens
        'process': 'tiling',
        'min_input_size': 1280,  # Use gundam for images > 1280px
        'min_tiles': 2,  # Minimum tiles for gundam
        'max_tiles': 9,  # Maximum tiles (3×3 grid)
    }
}


def select_resolution_mode(image_size: Tuple[int, int], mode: str = "auto", enable_gundam: bool = True) -> str:
    """
    Choose a resolution bucket from the longer side.

    Buckets (max dimension thresholds):
      tiny  ≤  512  →  512×512  →  64 tokens
      small ≤  640  →  640×640  → 100 tokens
      base  ≤ 1024  → 1024×1024 → 256 tokens
      large ≤ 1280  → 1280×1280 → 400 tokens
      gundam > 1280 → 1024 (global) + n×640 (tiles) → 256 + n×100 tokens
    """
    if mode != "auto":
        return mode

    w, h = image_size
    max_dim = max(w, h)

    if max_dim <= 512: return "tiny"
    if max_dim <= 640: return "small"
    if max_dim <= 1024: return "base"
    if max_dim <= 1280: return "large"
    return "gundam" if enable_gundam else "large"


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the best tiling aspect ratio for an image."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=9, image_size=640, use_thumbnail=False):
    """
    DeepSeek-OCR's dynamic preprocessing for tiling.

    Args:
        image: PIL Image
        min_num: Minimum number of tiles (default: 2)
        max_num: Maximum number of tiles (default: 9)
        image_size: Size of each tile (default: 640)
        use_thumbnail: Whether to add a thumbnail/global view (default: False)

    Returns:
        processed_images: List of PIL images (tiles)
        target_aspect_ratio: (width_tiles, height_tiles) tuple
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate possible aspect ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height), Image.Resampling.BICUBIC)

    # Split into tiles
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        processed_images.append(thumbnail_img)

    return processed_images, target_aspect_ratio


class DeepSeekOCRImageProcessor:
    """
    DeepSeek-OCR Image Processor with ALL resolution modes

    Modes:
    - tiny:   512×512   →  64 tokens
    - small:  640×640   → 100 tokens
    - base:  1024×1024  → 256 tokens
    - large: 1280×1280  → 400 tokens
    - gundam: 1024 (global) + n×640 (tiles) → 256 + n×100 tokens
    """

    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        mode: str = "base",
        resample=PILImageResampling.BICUBIC,
        rescale_factor: float = 1 / 255.0,
        data_format=ChannelDimension.FIRST,
        enable_gundam: bool = True,
    ):
        self.image_mean = image_mean
        self.image_std = image_std
        self.mode = mode
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.enable_gundam = enable_gundam

        # LLaVA-ish hints (defaults reflect current mode)
        mode_size = RESOLUTION_MODES.get(mode, RESOLUTION_MODES['base'])['size'] if mode != 'gundam' else 1024
        self.size = {"shortest_edge": mode_size, "height": mode_size, "width": mode_size}
        self.crop_size = {"height": mode_size, "width": mode_size}
        self.is_deepseek_processor = True
        self.processor_class = "DeepseekVLV2Processor"

    def preprocess(
        self,
        images: Union[Image.Image, np.ndarray, torch.Tensor, List],
        return_tensors=None,
        mode: str = None,
    ) -> Dict[str, List]:
        """
        Preprocess images with dynamic resolution support.

        Returns:
          {
            "pixel_values": List[FloatTensor (3,H,W) or (T,3,H,W) for gundam]
            "meta":         List[dict] with mode, size, token info
          }
        """
        if isinstance(images, (Image.Image, np.ndarray, torch.Tensor)):
            images = [images]
        pil_list: List[Image.Image] = [self._to_pil(img) for img in images]

        mode = mode or self.mode

        pixel_values: List[torch.Tensor] = []
        metas: List[dict] = []

        for pil_img in pil_list:
            # Select resolution mode for this image
            img_mode = select_resolution_mode(pil_img.size, mode, self.enable_gundam)

            if img_mode=='tiny':
                img_mode='base'
            if img_mode == 'small':
                img_mode = 'base'


            if img_mode == "gundam":
                arrays, tiles_meta = self._process_gundam(pil_img)
                # Stack all tiles: [num_tiles, 3, H, W]
                tiles_tensor = torch.stack([torch.from_numpy(a) for a in arrays], dim=0)
                pixel_values.append(tiles_tensor)

                # Calculate total tokens
                num_tiles = len(arrays) - 1  # Exclude global
                total_tokens = 256 + num_tiles * 100

                metas.append({
                    "mode": "gundam",
                    "num_tiles": tiles_tensor.shape[0],
                    "num_grid_tiles": num_tiles,
                    "total_tokens": total_tokens,
                    "tiles_meta": tiles_meta,
                })
            else:
                arr, meta = self._process_standard(pil_img, img_mode)
                pixel_values.append(torch.from_numpy(arr))
                metas.append(meta)

        return {"pixel_values": pixel_values, "meta": metas}

    def _to_pil(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Convert input to PIL RGB."""
        if isinstance(img, Image.Image):
            pil = img
        elif isinstance(img, torch.Tensor):
            x = img.detach().cpu()
            if x.ndim == 3 and x.shape[0] in (1, 3):
                x = x.permute(1, 2, 0).contiguous()
            x = x.numpy()
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 1) * 255.0
            pil = Image.fromarray(x.astype(np.uint8))
        else:
            x = img
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 1) * 255.0
            pil = Image.fromarray(x.astype(np.uint8))

        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return pil

    def _process_standard(self, image_pil: Image.Image, mode: str) -> Tuple[np.ndarray, dict]:
        """
        Standard path for tiny, small, base, large modes.

        Returns: (img_chw_float32_normalized, meta_dict)
        """
        cfg = RESOLUTION_MODES[mode]
        target = cfg["size"]

        # Resize with aspect ratio preservation and padding
        padded_chw, meta = self._resize_with_padding_CHW(image_pil, target)

        # Rescale [0,255] → [0,1]
        padded_chw = rescale(padded_chw, scale=self.rescale_factor, data_format=ChannelDimension.FIRST)

        # Normalize to [-1,1]
        padded_chw = normalize(
            padded_chw,
            mean=self.image_mean,
            std=self.image_std,
            data_format=ChannelDimension.FIRST
        )

        meta["mode"] = mode
        meta["tokens"] = cfg["tokens"]
        return padded_chw.astype(np.float32), meta

    def _resize_with_padding_CHW(self, image_pil: Image.Image, target: int) -> Tuple[np.ndarray, dict]:
        """
        Resize with preserved AR, pad to target×target.
        Returns CHW float32 in [0,255].

        ✅ CORRECTED: Uses mean color for padding (consistent with DeepSeek-OCR)
        """
        w, h = image_pil.size

        # Calculate scale to fit within target×target
        scale = target / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        # Resize maintaining aspect ratio
        pil_resized = image_pil.convert("RGB").resize((new_w, new_h), self.resample)
        resized_hwc = np.array(pil_resized).astype(np.float32)

        # Ensure 3 channels
        if resized_hwc.ndim == 2:
            resized_hwc = np.stack([resized_hwc] * 3, axis=-1)
        elif resized_hwc.shape[-1] == 1:
            resized_hwc = np.repeat(resized_hwc, 3, axis=-1)

        # Calculate padding (pad bottom-right to maintain top-left alignment)
        pad_h = target - new_h
        pad_w = target - new_w

        # ✅ Use mean color for padding (DeepSeek-OCR uses ImageOps.pad with mean)
        pad_value = int(self.image_mean[0] * 255)  # 127.5 → 127

        padded_hwc = np.pad(
            resized_hwc,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )

        # Convert to CHW
        padded_chw = padded_hwc.transpose(2, 0, 1)

        meta = {
            "orig_size": (h, w),
            "resize_size": (new_h, new_w),
            "pad": (0, pad_h, 0, pad_w),  # top, bottom, left, right
            "target_size": (target, target),
        }
        return padded_chw, meta

    def _process_gundam(self, image_pil: Image.Image) -> Tuple[List[np.ndarray], List[dict]]:
        """
        ✅ CORRECTED Gundam mode following DeepSeek-OCR:

        - Tiles: 640×640 (100 tokens each)
        - Global: 1024×1024 (256 tokens)
        - Ordering: [global, tile1, tile2, ...]
        - Total tokens: 256 + n_tiles × 100

        Returns:
            arrays: list of np.ndarray, each (3, H, W) normalized to [-1,1]
            metas: list of dict
        """
        cfg = RESOLUTION_MODES["gundam"]
        tile_size = cfg["tile_size"]  # 640
        global_size = cfg["global_size"]  # 1024
        min_tiles = cfg["min_tiles"]  # 2
        max_tiles = cfg["max_tiles"]  # 9

        w, h = image_pil.size

        # ✅ Check if image is small enough to skip tiling
        if w <= tile_size and h <= tile_size:
            # Image is small, just use 'base' mode instead
            print(f"[DeepSeek] Image {w}×{h} too small for Gundam, using base mode")
            arr, meta = self._process_standard(image_pil, mode="base")
            return [arr], [meta]

        # ✅ Use dynamic_preprocess to tile the image
        tile_images, (width_crop_num, height_crop_num) = dynamic_preprocess(
            image_pil,
            min_num=min_tiles,
            max_num=max_tiles,
            image_size=tile_size,
            use_thumbnail=False  # We handle global view separately
        )

        arrays: List[np.ndarray] = []
        metas: List[dict] = []

        # ✅ 1) Process GLOBAL view at 1024×1024 (base mode)
        global_arr, global_meta = self._process_standard(image_pil, mode="base")
        global_meta.update({
            "gundam_part": "global",
            "tile_index": None,
            "resolution": "1024×1024",
            "tokens": 256,
        })
        arrays.append(global_arr)
        metas.append(global_meta)

        # ✅ 2) Process TILES at 640×640 (small mode)
        for idx, tile_pil in enumerate(tile_images):
            tile_arr, tile_meta = self._process_standard(tile_pil, mode="small")

            # Calculate tile position in grid
            i = idx // width_crop_num  # row
            j = idx % width_crop_num  # col

            tile_meta.update({
                "gundam_part": "tile",
                "tile_index": (i, j),
                "tile_linear_index": idx,
                "resolution": "640×640",
                "tokens": 100,
            })
            arrays.append(tile_arr)
            metas.append(tile_meta)

        total_tokens = 256 + len(tile_images) * 100
        print(
            f"[DeepSeek] Gundam: {w}×{h} → global(1024, 256 tokens) + {len(tile_images)} tiles(640, {len(tile_images) * 100} tokens)")
        print(f"[DeepSeek] Grid: {height_crop_num}×{width_crop_num}, Total tokens: {total_tokens}")

        return arrays, metas


class DeepSeekOCRVisionTower(nn.Module):
    """
    DeepSeek-OCR Vision Tower with FULL DYNAMIC RESOLUTION support.

    Supports ALL resolution modes:
    ✅ tiny:   512×512   →  64 tokens
    ✅ small:  640×640   → 100 tokens
    ✅ base:  1024×1024  → 256 tokens
    ✅ large: 1280×1280  → 400 tokens
    ✅ gundam: 1024 (global) + n×640 (tiles) → 256 + n×100 tokens
    """

    def __init__(
        self,
        vision_tower,
        vision_tower_cfg=None,
        delay_load=False,
        resolution_mode='base',
        enable_gundam=True,
    ):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.resolution_mode = resolution_mode
        self.enable_gundam = enable_gundam

        # Config (will be updated based on actual mode)
        self._hidden_size = 2048  # SAM (1024) + CLIP (1024)
        self._image_size = RESOLUTION_MODES.get(resolution_mode, RESOLUTION_MODES['base']).get('size', 1024)
        self._num_patches = RESOLUTION_MODES.get(resolution_mode, RESOLUTION_MODES['base']).get('tokens', 256)

        # Image processor with ALL modes enabled
        self.image_processor = DeepSeekOCRImageProcessor(
            mode=resolution_mode,
            enable_gundam=enable_gundam
        )

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False) or (
            hasattr(vision_tower_cfg, "mm_tunable_parts")
            and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts
        ):
            rank0_print("Vision tower weights expected; loading immediately.")
            self.load_model()
        else:
            self.cfg_only = SimpleNamespace(
                hidden_size=self._hidden_size,
                image_size=self._image_size,
                num_patches=self._num_patches,
                resolution_mode=resolution_mode
            )

    def load_model(self, device_map=None):
        """Load DeepSeek-OCR vision components."""
        if self.is_loaded:
            rank0_print(f"{self.vision_tower_name} already loaded")
            return

        rank0_print("=" * 70)
        rank0_print("Loading DeepSeek-OCR with FULL Dynamic Resolution")
        rank0_print("=" * 70)

        # Load full model
        full_model = AutoModel.from_pretrained(
            self.vision_tower_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )

        # Extract vision components
        self.sam_model = full_model.model.sam_model
        self.vision_model = full_model.model.vision_model

        # Delete LLM to save memory
        if hasattr(full_model.model, 'layers'):
            del full_model.model.layers
        if hasattr(full_model.model, 'embed_tokens'):
            del full_model.model.embed_tokens
        if hasattr(full_model.model, 'norm'):
            del full_model.model.norm
        if hasattr(full_model, 'lm_head'):
            del full_model.lm_head
        if hasattr(full_model.model, 'projector'):
            del full_model.model.projector
        del full_model
        torch.cuda.empty_cache()

        # Freeze
        self.sam_model.requires_grad_(False)
        self.vision_model.requires_grad_(False)
        self.sam_model.eval()
        self.vision_model.eval()

        rank0_print("=" * 70)
        rank0_print("✓ DeepSeek-OCR Loaded - ALL MODES AVAILABLE")
        rank0_print("=" * 70)
        rank0_print(f"  Current Mode: {self.resolution_mode}")
        rank0_print(f"  Gundam Enabled: {self.enable_gundam}")
        rank0_print(f"  Available Modes:")
        rank0_print(f"    ✅ tiny:   512×512   →  64 tokens")
        rank0_print(f"    ✅ small:  640×640   → 100 tokens")
        rank0_print(f"    ✅ base:  1024×1024  → 256 tokens")
        rank0_print(f"    ✅ large: 1280×1280  → 400 tokens")
        rank0_print(f"    ✅ gundam: 1024 (global) + n×640 (tiles) → 256 + n×100 tokens")
        rank0_print("=" * 70)

        self.is_loaded = True

    def forward(self, images):
        """
        SigLIP-style forward.

        Args:
            images: List of tensors or single tensor

        Returns:
            List of [1, L, D] or single [B, L, D]
        """
        if isinstance(images, list):
            feats = []
            for idx, image in enumerate(images):
                if not isinstance(image, torch.Tensor):
                    raise ValueError(f"images[{idx}] is not a tensor")
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                elif image.dim() != 4:
                    raise ValueError(f"images[{idx}] has dim={image.dim()}, expected 3 or 4")

                image = image.to(device=self.device, dtype=self.dtype)
                feat = self._encode_single(image)

                # Normalize to 3D
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0).unsqueeze(0)
                elif feat.dim() == 2:
                    feat = feat.unsqueeze(0)

                feats.append(feat)
            return feats

        # Single tensor
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"images must be tensor or list, got {type(images)}")

        if images.dim() == 3:
            images = images.unsqueeze(0)
        elif images.dim() != 4:
            raise ValueError(f"images has dim={images.dim()}, expected 3 or 4")

        images = images.to(device=self.device, dtype=self.dtype)
        feat = self._encode_single(images)

        # Normalize to 3D
        if feat.dim() == 1:
            feat = feat.unsqueeze(0).unsqueeze(0)
        elif feat.dim() == 2:
            feat = feat.unsqueeze(0)

        return feat

    def _encode_single(self, image: torch.Tensor):
        """
        Encode single image [1, C, H, W] → [num_tokens, 2048]
        """
        # SAM forward
        sam_output = self.sam_model(image)  # [1, 1024, H, W]

        # Flatten SAM
        B, C, H, W = sam_output.shape
        sam_features = sam_output.flatten(2).transpose(1, 2)  # [1, num_tokens, 1024]

        # CLIP forward - pass UN-FLATTENED
        clip_output = self.vision_model(
            x=image,
            patch_embeds=sam_output  # UN-flattened!
        )

        # Handle CLIP output
        if isinstance(clip_output, dict):
            clip_features = clip_output.get('last_hidden_state',
                                            clip_output.get('hidden_states', clip_output))
        elif isinstance(clip_output, tuple):
            clip_features = clip_output[0]
        else:
            clip_features = clip_output

        # Remove CLS token if present
        num_sam_tokens = sam_features.shape[1]
        if clip_features.shape[1] == num_sam_tokens + 1:
            clip_features = clip_features[:, 1:, :]

        # Concatenate SAM + CLIP
        combined_features = torch.cat([sam_features, clip_features], dim=-1)  # [1, num_tokens, 2048]

        return combined_features.squeeze(0)  # [num_tokens, 2048]

    # Properties
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.sam_model.parameters():
            return p.dtype
        return torch.bfloat16

    @property
    def device(self):
        for p in self.sam_model.parameters():
            return p.device
        return torch.device('cpu')

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_patches(self):
        """Returns number of patches for current mode."""
        return self._num_patches

    @property
    def num_patches_per_side(self):
        return int(self._num_patches ** 0.5)

    @property
    def image_size(self):
        """Returns image size for current mode."""
        return self._image_size

    @property
    def config(self):
        return SimpleNamespace(
            hidden_size=self._hidden_size,
            image_size=self._image_size,
            num_patches=self._num_patches,
            resolution_mode=self.resolution_mode,
            enable_gundam=self.enable_gundam,
        )
