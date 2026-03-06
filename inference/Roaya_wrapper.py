#!/usr/bin/env python3
"""
RoayaVL Wrapper
======================================================================

"""

import json
import copy
from typing import List, Optional, Tuple, Union
from pathlib import Path

import torch
from PIL import Image

# Register qwen_2_5 as an alias for qwen2 (needed for configs that use model_type="qwen_2_5")
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

if "qwen_2_5" not in CONFIG_MAPPING:
    CONFIG_MAPPING.register("qwen_2_5", Qwen2Config)
    print("✓ Registered qwen_2_5 → qwen2")

# Under the hood, you still use llava runtime (loader + processing)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle


import warnings

warnings.filterwarnings(
    "ignore",
    message=".*copying from a non-meta parameter.*to a meta parameter.*")


def _maybe_patch_config_paths(
    model_dir: Union[str, Path],
    patch_mm_vision_tower: Optional[str] = None,
    patch_name_or_path: Optional[str] = None,
    inplace: bool = False,
) -> None:
    """Patch absolute paths inside config.json (useful after copying checkpoints)."""
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return

    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        print(f"[WARN] Could not parse config.json at: {cfg_path}")
        return

    changed = False

    if patch_mm_vision_tower is not None and cfg.get("mm_vision_tower") != patch_mm_vision_tower:
        cfg["mm_vision_tower"] = patch_mm_vision_tower
        changed = True

    if patch_name_or_path is not None and cfg.get("_name_or_path") != patch_name_or_path:
        cfg["_name_or_path"] = patch_name_or_path
        changed = True

    if changed:
        if inplace:
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
            print(f"[PATCH] Updated config.json (in-place): {cfg_path}")
        else:
            print("[WARN] config.json paths differ; set patch_config_inplace=True to rewrite.")


class RoayaVLWrapper:
    """
    RoayaVL wrapper around the llava runtime, with stable loading regardless of folder name.
    """

    def __init__(
        self,
        model_path: str,
        model_base: Optional[str] = None,
        device: str = "cuda",
        conv_template: str = "qwen_2_5",
        attn_implementation: str = "eager",
        # INTERNAL routing string for llava loader (does NOT depend on your folder name)
        # Keep 'llava' + 'qwen' here so the loader never picks LlavaLlama by mistake.
        roayvl_backend: str = "qwen2_5",
        # Optional patching after copying
        patch_mm_vision_tower: Optional[str] = None,
        patch_name_or_path: Optional[str] = None,
        patch_config_inplace: bool = False,
        # Optional dtype control for vision tensors
        vision_dtype: Optional[torch.dtype] = None,
        # Fail fast if wrong backend is loaded
        strict_qwen_backend: bool = True,
        # Print extra debug
        verbose: bool = True,
    ):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.conv_template = conv_template
        self.model_path = str(model_path)
        self.model_base = str(model_base) if model_base is not None else None

        # Build an internal loader routing string (NOT your folder name)
        # This is the only place we "mention" llava.
        self._builder_model_name = f"roayvl_llava_{roayvl_backend}".lower()

        _maybe_patch_config_paths(
            model_dir=model_path,
            patch_mm_vision_tower=patch_mm_vision_tower,
            patch_name_or_path=patch_name_or_path,
            inplace=patch_config_inplace,
        )

        if verbose:
            print(f"[ROAYVL] model_path={self.model_path}")
            print(f"[ROAYVL] model_base={self.model_base}")
            print(f"[ROAYVL] device={self.device}")
            print(f"[ROAYVL] internal_loader_tag={self._builder_model_name}  (folder name not used)")

        # ✅ Folder-name independent loading
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path=self.model_path,
            model_base=self.model_base,
            model_name=self._builder_model_name,
            device_map=self.device,
            attn_implementation=attn_implementation,
        )

        self.model.eval()
        self.model.tie_weights()

        # Vision dtype
        if vision_dtype is not None:
            self.vision_dtype = vision_dtype
        else:
            if self.device != "cpu":
                model_dtype = getattr(self.model, "dtype", torch.float16)
                self.vision_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
            else:
                self.vision_dtype = torch.float32

        if verbose:
            print("[ROAYVL] Model Class:", self.model.__class__.__name__)
            print("[ROAYVL] image_processor:", self.image_processor)
            print("[ROAYVL] mm_vision_tower:", getattr(self.model.config, "mm_vision_tower", None))
            print("[ROAYVL] model_dtype:", getattr(self.model, "dtype", None))
            print("[ROAYVL] vision_dtype:", self.vision_dtype)

        # Fail fast if wrong backend
        if strict_qwen_backend:
            cls = self.model.__class__.__name__.lower()
            if "qwen" not in cls:
                raise RuntimeError(
                    f"[ROAYVL] Wrong backend class loaded: {self.model.__class__.__name__}\n"
                    f"Expected a Qwen-based LLaVA model.\n"
                    f"Internal loader tag used: {self._builder_model_name}\n"
                    f"Tip: keep roayvl_backend='qwen2_5' (default)."
                )

        # Ensure image_token_index exists
        if not hasattr(self.model.config, "image_token_index") or self.model.config.image_token_index is None:
            self.model.config.image_token_index = IMAGE_TOKEN_INDEX
            if verbose:
                print(f"[ROAYVL] [FIX] set config.image_token_index = {IMAGE_TOKEN_INDEX}")

        if hasattr(self.model.config, "image_aspect_ratio") and verbose:
            print(f"[ROAYVL] image_aspect_ratio = {self.model.config.image_aspect_ratio}")

        # Warn if vision tower path is missing
        mm_tower = getattr(self.model.config, "mm_vision_tower", None)
        if isinstance(mm_tower, str) and mm_tower and not Path(mm_tower).exists():
            print(f"[ROAYVL] [WARN] mm_vision_tower path does not exist: {mm_tower}")
            print("        Use patch_mm_vision_tower=... (optionally patch_config_inplace=True).")

    def _repeat_image_tokens(self, n: int) -> str:
        return (DEFAULT_IMAGE_TOKEN * n) + ("\n" if n > 0 else "")

    def _build_prompt(self, text: str, n_images: int, conv_template: Optional[str] = None):
        prefix = self._repeat_image_tokens(n_images)
        user_msg = prefix + (text or "")

        conv_name = conv_template or self.conv_template
        if conv_name not in conv_templates:
            raise ValueError(f"Unknown conv_template='{conv_name}'. Available: {list(conv_templates.keys())}")

        conv = copy.deepcopy(conv_templates[conv_name])
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt(), conv

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        images: Optional[List[Union[str, Path, Image.Image]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        conv_template: Optional[str] = None,
    ) -> str:
        image_inputs = images or []

        # Convert to PIL
        pil_images: List[Image.Image] = []
        for img in image_inputs:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

        # Prompt
        prompt_text, conv = self._build_prompt(text, len(pil_images), conv_template=conv_template)

        # Images -> tensors
        image_tensors = None
        image_sizes: Optional[List[Tuple[int, int]]] = None
        if pil_images:
            image_sizes = [im.size for im in pil_images]  # (W,H)
            image_tensors = process_images(pil_images, self.image_processor, self.model.config)
            image_tensors = [t.to(dtype=self.vision_dtype, device=self.device) for t in image_tensors]

        # Tokenize with IMAGE_TOKEN_INDEX mapping
        input_ids = tokenizer_image_token(
            prompt_text,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        # Sanity check: image token count
        n_tokens = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
        if n_tokens != len(pil_images):
            print(f"[ROAYVL] [WARN] image token count ({n_tokens}) != images provided ({len(pil_images)})")

        # Generation flags (avoid temperature warnings)
        do_sample = temperature > 0

        out_ids = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            max_new_tokens=max_new_tokens,
        )

        out_text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

        # Strip template artifacts
        if conv.sep_style == SeparatorStyle.TWO:
            out_text = out_text.split(conv.sep2)[-1].strip()
        elif conv.sep_style == SeparatorStyle.LLAMA_2:
            out_text = out_text.split("[/INST]")[-1].strip()

        return out_text

    @torch.inference_mode()
    def generate_text_only(self, text: str, max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        return self.generate(text=text, images=None, max_new_tokens=max_new_tokens, temperature=temperature)

    def __call__(self, text: str, images: Optional[List[Union[str, Path, Image.Image]]] = None, **kwargs) -> str:
        return self.generate(text, images, **kwargs)

