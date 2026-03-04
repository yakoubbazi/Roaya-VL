"""
DeepSeek-OCR Vision Tower with DYNAMIC RESOLUTION + proper Gundam tiling

Supports 5 modes from the paper:
- Tiny  (512×512,  64 tokens)
- Small (640×640, 100 tokens)
- Base  (1024×1024, 256 tokens)
- Large (1280×1280, 400 tokens)
- Gundam (dynamic tiling, n×100 + 256 tokens: 640×640 local tiles + 1024×1024 global)

This version:
- Uses DeepSeek's original dynamic_preprocess() to generate local tiles.
- Global view is 1024×1024 (Base).
- Local tiles are 640×640 (Small).
"""

from typing import Optional, List, Tuple, Dict, Union
from types import SimpleNamespace
import math
import numpy as np
from PIL import Image
from PIL import Image as PILImage
from transformers.image_transforms import (
    convert_to_rgb,
    rescale,
    normalize,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import ChannelDimension, PILImageResampling
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from llava.utils import rank0_print

# ============================================================
# Resolution Modes
# ============================================================

RESOLUTION_MODES = {
    "tiny": {
        "size": 512,
        "tokens": 64,
        "process": "padding",
        "max_input_size": 640,
    },
    "small": {
        "size": 640,
        "tokens": 100,
        "process": "padding",
        "max_input_size": 768,
    },
    "base": {
        "size": 1024,
        "tokens": 256,
        "process": "padding",
        "max_input_size": 1280,
    },
    "large": {
        "size": 1280,
        "tokens": 400,
        "process": "padding",
        "max_input_size": 1536,
    },
    # Gundam tiling mode (DeepSeek-style)
    "gundam": {
        "tile_size": 640,       # local tile size (pre-SAM/CLIP)
        "global_size": 1024,    # global view size
        "tile_tokens": 100,
        "global_tokens": 256,
        "process": "tiling",
        "min_input_size": 1536,  # threshold to enable Gundam in auto mode
    },
}


def select_resolution_mode(
    image_size: Tuple[int, int],
    mode: str = "auto",
    enable_gundam: bool = True
) -> str:
    """
    Choose a resolution bucket from the longer side.

    Buckets (longer side thresholds):
      tiny   ≤  512
      small  ≤  640
      base   ≤ 1024
      large  ≤ 1280
      gundam > 1280 (ONLY if enable_gundam=True)
    """
    if mode != "auto":
        return mode

    w, h = image_size
    max_dim = max(w, h)

    if max_dim <= 512:
        return "tiny"
    if max_dim <= 640:
        return "small"
    if max_dim <= 1024:
        return "base"
    if max_dim <= 1280:
        return "large"

    # Very large images: optionally switch to Gundam
    if enable_gundam:
        return "gundam"
    return "large"


# ============================================================
# DeepSeek dynamic_preprocess (original tiling logic)
# ============================================================

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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
            # tie-breaker on area condition
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: PILImage,
    min_num: int = 2,
    max_num: int = 9,
    image_size: int = 640,
    use_thumbnail: bool = False
):
    """
    DeepSeek-style dynamic tiling:

      - Pick (i, j) grid shape with i*j in [min_num, max_num]
        whose aspect ratio i/j is closest to the image AR.
      - Resize once to (image_size*i, image_size*j).
      - Crop into i*j non-overlapping tiles of size (image_size, image_size).

    Returns:
      processed_images: List[PIL.Image] each with size (image_size, image_size)
      target_aspect_ratio: (i, j)
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images, target_aspect_ratio


# ============================================================
# Image Processor
# ============================================================

class DeepSeekOCRImageProcessor:
    """
    DeepSeek-OCR style image processor.

    - Output: list of tensors (3, H, W) normalized to [-1,1]
      using mean=std=0.5 after rescaling to [0,1].
    - Aspect ratio preserved via padding for standard modes.
    - Gundam mode uses dynamic_preprocess for local tiles (640×640)
      + global view (1024×1024).
    """

    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        mode: str = "base",  # 'auto' | 'tiny' | 'small' | 'base' | 'large' | 'gundam'
        resample=PILImageResampling.BICUBIC,
        rescale_factor: float = 1 / 255.0,  # [0,255] → [0,1]
        data_format=ChannelDimension.FIRST,  # CHW
        enable_gundam: bool = True,
        max_gundam_tiles: int = 9,  # maximum (global + locals)
    ):
        self.image_mean = image_mean
        self.image_std = image_std
        self.mode = mode
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.enable_gundam = enable_gundam
        self.max_gundam_tiles = max_gundam_tiles

        # LLaVA-ish hints (default to Base)
        self.size = {"shortest_edge": 1024, "height": 1024, "width": 1024}
        self.crop_size = {"height": 1024, "width": 1024}
        self.is_deepseek_processor = True
        self.processor_class = "DeepseekVLV2Processor"

    # --------------------------------
    # Public API
    # --------------------------------
    def preprocess(
        self,
        images: Union[PILImage.Image, np.ndarray, torch.Tensor, List],
        return_tensors=None,  # purposely ignored (we always return lists)
        mode: str = None,
    ) -> Dict[str, List]:
        """
        Returns:
          {
            "pixel_values": List[Tensor]   # each (3,H,W) or (T,3,H,W) for Gundam
            "meta":         List[dict]
          }
        """
        if isinstance(images, (PILImage.Image, np.ndarray, torch.Tensor)):
            images = [images]

        pil_list: List[PILImage.Image] = [self._to_pil(img) for img in images]
        mode = mode or self.mode

        pixel_values: List[torch.Tensor] = []
        metas: List[dict] = []

        for pil_img in pil_list:
            img_mode = select_resolution_mode(
                pil_img.size, mode=mode, enable_gundam=self.enable_gundam
            )

            if img_mode=='tiny':
                img_mode='base'

            if img_mode=='small':
                img_mode='base'

            #if img_mode=='large':
            #    img_mode='base'
            #if img_mode == 'gundam':
            #    img_mode = 'base'

            if img_mode == "gundam":
                arrays, tiles_meta = self._process_gundam(pil_img)
                # arrays: list of np.ndarray (3,H,W)
                tiles_tensor = torch.stack(
                    [torch.from_numpy(a) for a in arrays], dim=0
                )  # [T,3,H,W]
                pixel_values.append(tiles_tensor)
                metas.append(
                    {
                        "mode": "gundam",
                        "num_tiles": tiles_tensor.shape[0],
                        "tiles_meta": tiles_meta,
                    }
                )
            else:
                arr, meta = self._process_standard(pil_img, img_mode)
                pixel_values.append(torch.from_numpy(arr))
                metas.append(meta)

        return {"pixel_values": pixel_values, "meta": metas}

    # --------------------------------
    # Internals
    # --------------------------------
    def _to_pil(self, img: Union[PILImage.Image, np.ndarray, torch.Tensor]) -> PILImage.Image:
        """Convert input (PIL | np | torch) to PIL RGB."""
        if isinstance(img, PILImage.Image):
            pil = img
        elif isinstance(img, torch.Tensor):
            x = img.detach().cpu()
            if x.ndim == 3 and x.shape[0] in (1, 3):  # CHW → HWC
                x = x.permute(1, 2, 0).contiguous()
            x = x.numpy()
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 1) * 255.0
            pil = PILImage.fromarray(x.astype(np.uint8))
        else:  # np.ndarray
            x = img
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 1) * 255.0
            pil = PILImage.fromarray(x.astype(np.uint8))

        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return pil

    def _process_standard(self, image_pil: PILImage.Image, mode: str) -> Tuple[np.ndarray, dict]:
        """
        Standard path (no Gundam tiling).
        Returns:
          (img_chw_float32_normalized, meta_dict)
        """
        cfg = RESOLUTION_MODES[mode]
        target = cfg["size"]

        hwc = np.array(convert_to_rgb(image_pil), dtype=np.uint8)

        if cfg.get("process", "padding") == "resize":
            chw = to_channel_dimension_format(hwc, ChannelDimension.FIRST)
            chw = resize(
                chw,
                size=(target, target),
                resample=self.resample,
                data_format=ChannelDimension.FIRST,
            )
            chw = rescale(
                chw,
                scale=self.rescale_factor,
                data_format=ChannelDimension.FIRST,
            )
            chw = normalize(
                chw,
                mean=self.image_mean,
                std=self.image_std,
                data_format=ChannelDimension.FIRST,
            )
            meta = {
                "orig_size": (image_pil.height, image_pil.width),
                "resize_size": (target, target),
                "pad": (0, 0, 0, 0),
                "target_size": (target, target),
                "mode": mode,
            }
            return chw.astype(np.float32), meta

        # padding mode
        padded_chw, meta = self._resize_with_padding_CHW(image_pil, target)
        padded_chw = rescale(
            padded_chw,
            scale=self.rescale_factor,
            data_format=ChannelDimension.FIRST,
        )
        padded_chw = normalize(
            padded_chw,
            mean=self.image_mean,
            std=self.image_std,
            data_format=ChannelDimension.FIRST,
        )
        meta["mode"] = mode
        return padded_chw.astype(np.float32), meta

    def _resize_with_padding_CHW(
        self,
        image_pil: PILImage.Image,
        target: int,
    ) -> Tuple[np.ndarray, dict]:
        """Resize with preserved AR, pad to target×target, return CHW float32 [0,255]."""
        w, h = image_pil.size
        scale = target / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        pil = image_pil.convert("RGB").resize((new_w, new_h), self.resample)
        resized_hwc = np.array(pil).astype(np.float32)

        if resized_hwc.ndim == 2:
            resized_hwc = np.stack([resized_hwc] * 3, axis=-1)
        elif resized_hwc.shape[-1] == 1:
            resized_hwc = np.repeat(resized_hwc, 3, axis=-1)

        pad_h = target - new_h
        pad_w = target - new_w
        padded_hwc = np.pad(
            resized_hwc,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        padded_chw = padded_hwc.transpose(2, 0, 1)
        meta = {
            "orig_size": (h, w),
            "resize_size": (new_h, new_w),
            "pad": (0, pad_h, 0, pad_w),  # top, bottom, left, right
            "target_size": (target, target),
        }
        return padded_chw, meta

    def _process_gundam(
        self,
        image_pil: PILImage.Image,
    ) -> Tuple[List[np.ndarray], List[dict]]:
        """
        DeepSeek 'Gundam' mode:

          - Local tiles:
              use dynamic_preprocess(image, image_size=640)
              → Nx 640×640 tiles (2 ≤ N ≤ 9).
          - Global tile:
              1024×1024, processed with 'base' mode.
          - Total tiles is capped by self.max_gundam_tiles
            (including global).

        Returns:
            arrays: list of np.ndarray, each (3,H,W) in [-1,1]
            metas:  list of dict with per-tile metadata
        """
        cfg = RESOLUTION_MODES["gundam"]
        tile_size = 640 #cfg["tile_size"]      # 640
        global_size = 1024 #cfg["global_size"]  # 1024

        # 1) local tiles via dynamic_preprocess
        local_pils, (grid_w, grid_h) = dynamic_preprocess(
            image_pil,
            min_num=2,
            max_num=9,
            image_size=tile_size,
            use_thumbnail=False,
        )
        total_local = len(local_pils)

        keep_global = True
        max_tiles = self.max_gundam_tiles
        max_local_tiles = max_tiles - (1 if keep_global else 0)
        if max_local_tiles < 0:
            max_local_tiles = 0

        if total_local > max_local_tiles > 0:
            # uniform subsampling of local tiles
            stride = math.ceil(total_local / max_local_tiles)
            kept_indices = list(range(0, total_local, stride))[:max_local_tiles]
        else:
            kept_indices = list(range(total_local))

        arrays: List[np.ndarray] = []
        metas: List[dict] = []

        # 2) Global tile (1024×1024) – 'base' mode
        if keep_global:
            g_arr, g_meta = self._process_standard(image_pil, mode="base")
            g_meta.update(
                {
                    "gundam_part": "global",
                    "tile_index": None,
                    "tile_linear_index": None,
                    "grid_shape": None,
                }
            )
            arrays.append(g_arr)
            metas.append(g_meta)

        # 3) Local tiles (640×640) – 'small' mode
        # 3) Local tiles → use SAME padding style as standard, but to 1024
        target = 1024  # 1024

        for lin_idx in kept_indices:
            tile_pil = local_pils[lin_idx]

            # --- use your existing padding helper ---
            padded_chw, meta = self._resize_with_padding_CHW(
                tile_pil,
                target=target,
            )   # CHW in [0,255]

            # then exactly the same post-processing as _process_standard (padding branch)
            padded_chw = rescale(
                padded_chw,
                scale=self.rescale_factor,
                data_format=ChannelDimension.FIRST,
            )
            padded_chw = normalize(
                padded_chw,
                mean=self.image_mean,
                std=self.image_std,
                data_format=ChannelDimension.FIRST,
            )

            meta.update(
                {
                    "mode": "gundam_tile",
                    "gundam_part": "tile",
                    "tile_index": None,
                    "tile_linear_index": lin_idx,
                    "grid_shape": (grid_w, grid_h),
                    "target_size": (target, target),
                }
            )

            arrays.append(padded_chw.astype(np.float32))
            metas.append(meta)


        #original_local = total_local
        #kept_local = len(kept_indices)
        #print(
        #    f"[DeepSeek] Gundam: original local tiles={original_local}, "
        #    f"kept_local_tiles={kept_local}, max_tiles={self.max_gundam_tiles}"
        #)
        #print(
        #    f"[DeepSeek] Gundam: final tiles (incl. global={keep_global}) "
        #    f"= {len(arrays)}"
        #)
        #3if arrays:
        #    print(f"[DeepSeek] final tile[0] shape: {arrays[0].shape}")  # (3, H, W)

        return arrays, metas


# ============================================================
# Vision Tower with Dynamic Resolution
# ============================================================

class DeepSeekOCRVisionTower(nn.Module):
    """
    DeepSeek-OCR Vision Tower with DYNAMIC RESOLUTION support.

    - In LLaVA-style integration, you typically:
        * Use 'base' mode for fixed 1024×1024 training tokens (256 tokens).
        * Optionally enable Gundam for inference on huge images (tiles + global).

    - This module only handles the vision encoder (SAM + CLIP-like model).
    """

    def __init__(
        self,
        vision_tower,
        vision_tower_cfg=None,
        delay_load=False,
        resolution_mode: str = "base",
        pad_to_max: bool = False,
        enable_gundam: bool = True,
        max_gundam_tiles: int = 9,
        unfreeze_bottleneck: bool = False,  # ← ADD THIS
    ):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.resolution_mode = resolution_mode
        self.pad_to_max = pad_to_max
        self.unfreeze_bottleneck = unfreeze_bottleneck  # ← ADD THIS LINE!

        self._hidden_size = 2048  # SAM (1024) + CLIP (1024)
        self._image_size = 1024
        self._num_patches = 256   # 32×32

        self.image_processor = DeepSeekOCRImageProcessor(
            mode=resolution_mode,
            enable_gundam=enable_gundam,
            max_gundam_tiles=max_gundam_tiles,
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
                resolution_mode=resolution_mode,
            )

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print(f"{self.vision_tower_name} already loaded; skipping.")
            return

        rank0_print("=" * 70)
        rank0_print("Loading DeepSeek-OCR (SAM+CLIP) Vision Tower")
        rank0_print("=" * 70)

        full_model = AutoModel.from_pretrained(
            self.vision_tower_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )

        rank0_print("Extracting vision components...")
        self.sam_model = full_model.model.sam_model
        self.vision_model = full_model.model.vision_model

        # delete LLM parts
        rank0_print("Deleting LLM components...")
        if hasattr(full_model.model, "layers"):
            del full_model.model.layers
        if hasattr(full_model.model, "embed_tokens"):
            del full_model.model.embed_tokens
        if hasattr(full_model.model, "norm"):
            del full_model.model.norm
        if hasattr(full_model, "lm_head"):
            del full_model.lm_head
        if hasattr(full_model.model, "projector"):
            del full_model.model.projector
        del full_model
        torch.cuda.empty_cache()

        self.sam_model.requires_grad_(False)
        self.vision_model.requires_grad_(False)
        self.sam_model.eval()
        self.vision_model.eval()


        # Unfreeze bottleneck if requested
        if self.unfreeze_bottleneck:
            self.sam_model.neck.requires_grad_(True)
            self.sam_model.net_2.requires_grad_(True)
            self.sam_model.net_3.requires_grad_(True)
            rank0_print("🔥 Bottleneck layers TRAINABLE (~6M params)")

        self._hidden_size = 2048
        self._image_size = 1024
        self._num_patches = 256

        rank0_print("=" * 70)
        rank0_print("✓ DeepSeek-OCR Vision Tower Loaded")
        rank0_print("=" * 70)
        rank0_print(f"  Resolution Mode: {self.resolution_mode}")
        for name, cfg in RESOLUTION_MODES.items():
            if name != "gundam":
                rank0_print(
                    f"    - {name}: {cfg['size']}×{cfg['size']} → {cfg['tokens']} tokens"
                )
            else:
                rank0_print("    - gundam: dynamic tiles + global")
        rank0_print(f"  Hidden Size: {self._hidden_size}")
        rank0_print("=" * 70)

        self.is_loaded = True

    def forward(self, images):
        """
        SigLIP-style forward:

        - If `images` is a list of tensors:
            returns list of [1, L, D] tensors.
        - If `images` is a single tensor:
            returns [B, L, D].

        NOTE: This assumes images are already preprocessed (e.g., from image_processor)
              into CHW / BCHW shapes. Gundam tiling should be handled on the
              preprocessing side, where you call image_processor.preprocess().
        """
        if isinstance(images, list):
            feats = []
            for idx, image in enumerate(images):
                if not isinstance(image, torch.Tensor):
                    raise ValueError(f"images[{idx}] is not a tensor, got {type(image)}")
                if image.dim() == 3:
                    image = image.unsqueeze(0)  # [1,C,H,W]
                elif image.dim() == 4:
                    pass
                else:
                    raise ValueError(f"images[{idx}] has dim={image.dim()}, expected 3 or 4")

                image = image.to(device=self.device, dtype=self.dtype)
                feat = self._encode_single(image)  # [1,L,D]

                if feat.dim() == 2:
                    feat = feat.unsqueeze(0)
                feats.append(feat)
            return feats

        # single tensor
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"images is not a tensor or list, got {type(images)}")

        if images.dim() == 3:
            images = images.unsqueeze(0)
        elif images.dim() == 4:
            pass
        else:
            raise ValueError(f"images has dim={images.dim()}, expected 3 or 4")

        images = images.to(device=self.device, dtype=self.dtype)
        feat = self._encode_single(images)

        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        return feat  # [B,L,D]

    def _encode_single(self, image: torch.Tensor):
        """
        Encode a single (batch) of images.

        Args:
            image: [B, C, H, W]

        Returns:
            features: [B, L, 2048] (SAM 1024 + CLIP 1024)
        """
        sam_output = self.sam_model(image)  # [B,1024,H',W']
        B, C, H, W = sam_output.shape
        sam_features = sam_output.flatten(2).transpose(1, 2)  # [B, L, 1024]

        clip_output = self.vision_model(
            x=image,
            patch_embeds=sam_output,
        )
        if isinstance(clip_output, dict):
            clip_features = clip_output.get(
                "last_hidden_state",
                clip_output.get("hidden_states", clip_output),
            )
        elif isinstance(clip_output, tuple):
            clip_features = clip_output[0]
        else:
            clip_features = clip_output

        # remove CLS if present
        if clip_features.shape[1] == sam_features.shape[1] + 1:
            clip_features = clip_features[:, 1:, :]

        combined_features = torch.cat(
            [sam_features, clip_features], dim=-1
        )  # [B, L, 2048]
        return combined_features

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
        return torch.device("cpu")

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_patches(self):
        return self._num_patches

    @property
    def num_patches_per_side(self):
        return int(self._num_patches ** 0.5)

    @property
    def image_size(self):
        return self._image_size

    @property
    def config(self):
        return SimpleNamespace(
            hidden_size=self._hidden_size,
            image_size=self._image_size,
            num_patches=self._num_patches,
            resolution_mode=self.resolution_mode,
        )
