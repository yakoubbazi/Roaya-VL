#!/usr/bin/env python3
# roayavl.py
# Wrapper around your working LLaVA-NeXT/Qwen single-turn multi-image code
# - Loads model once
# - Stateless calls: roaya.chat(text, images=[...])
# - Uses SAME conv template ("qwen_2_5") and SAME decoding logic you used

import copy
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

# Register qwen_2_5 alias for qwen2 (same as your code)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

if "qwen_2_5" not in CONFIG_MAPPING:
    CONFIG_MAPPING.register("qwen_2_5", Qwen2Config)
    print("✓ Registered qwen_2_5 → qwen2")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle


def _repeat_image_tokens(n: int) -> str:
    return (DEFAULT_IMAGE_TOKEN * n) + ("\n" if n > 0 else "")


def _pad_images_to_same_size(img_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Optional helper (same as your function). Your working code currently does NOT call it.
    """
    if not img_tensors:
        return img_tensors

    norm = []
    max_h, max_w = 0, 0
    for t in img_tensors:
        if t.dim() == 3:
            t = t.unsqueeze(0)
        elif t.dim() == 4 and t.shape[0] != 1:
            norm.extend([ti.unsqueeze(0) if ti.dim() == 3 else ti for ti in t])
            continue
        _, _, h, w = t.shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        norm.append(t)

    padded = []
    for t in norm:
        _, _, h, w = t.shape
        pad_right = max_w - w
        pad_bottom = max_h - h
        if pad_right or pad_bottom:
            t = F.pad(t, (0, pad_right, 0, pad_bottom), mode="constant", value=0)
        padded.append(t)
    return padded


@dataclass
class RoayaVLParams:
    conv_template: str = "qwen_2_5"   # keep identical to your working script
    max_new_tokens: int = 512
    temperature: float = 0.0
    enforce_english: bool = False     # optional helper prompt
    pad_images: bool = False          # optional (your script leaves it off)


class RoayaVL:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model_path = model_path

        model_name = get_model_name_from_path(model_path)
        print(f"[LOAD] path={model_path}")
        print(f"[LOAD] name={model_name}")
        print(f"[LOAD] device={self.device}")

        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name or "llava_qwen",
            device_map=self.device,
            attn_implementation="eager",
        )
        self.model.eval()
        try:
            self.model.tie_weights()
        except Exception:
            pass

        # Ensure image_token_index (same as your code)
        if not hasattr(self.model.config, "image_token_index") or self.model.config.image_token_index is None:
            self.model.config.image_token_index = IMAGE_TOKEN_INDEX
            print(f"[FIX] set config.image_token_index = {IMAGE_TOKEN_INDEX}")
        else:
            print(f"[CFG] image_token_index = {self.model.config.image_token_index}")

        print("image_processor:", self.image_processor)
        print("mm_vision_tower:", getattr(self.model.config, "mm_vision_tower", None))
        print("vision_tower_type:", getattr(self.model.config, "vision_tower", None))
        print("image_aspect_ratio:", getattr(self.model.config, "image_aspect_ratio", None))

        self._lock = threading.Lock()

    @torch.inference_mode()
    def chat(
        self,
        text: str,
        images: Optional[List[str]] = None,
        params: Optional[RoayaVLParams] = None,
    ) -> str:
        params = params or RoayaVLParams()
        images = images or []

        # Optional helper instruction
        if params.enforce_english:
            text = "Please answer in English only. Do not output Chinese.\n" + (text or "")

        # 1) Build user message (identical behavior)
        user_msg = _repeat_image_tokens(len(images)) + (text or "")

        # 2) Conversation template (identical behavior)
        conv = copy.deepcopy(conv_templates[params.conv_template])
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # 3) Process images (identical behavior)
        image_tensors: List[torch.Tensor] = []
        image_sizes: List[Tuple[int, int]] = []

        if images:
            pil_images = [Image.open(p).convert("RGB") for p in images]
            image_sizes = [im.size for im in pil_images]  # (W,H)
            image_tensors = process_images(pil_images, self.image_processor, self.model.config)

            if self.device != "cpu":
                image_tensors = [t.to(dtype=torch.float16, device=self.device) for t in image_tensors]
            else:
                image_tensors = [t.to(dtype=torch.float32, device=self.device) for t in image_tensors]

            # Optional: only if you want it (your working script leaves it off)
            if params.pad_images:
                image_tensors = _pad_images_to_same_size(image_tensors)

        # 4) Tokenize (identical behavior)
        input_ids = tokenizer_image_token(
            prompt_text,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        # sanity check
        n_tokens = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
        if n_tokens != len(images):
            print(f"[WARN] image token count ({n_tokens}) != images provided ({len(images)})")

        # 5) Generate (identical behavior)
        with self._lock:
            out_ids = self.model.generate(
                input_ids,
                images=image_tensors if image_tensors else None,
                image_sizes=image_sizes if image_sizes else None,
                do_sample=False if params.temperature == 0 else True,
                temperature=params.temperature if params.temperature > 0 else 0.01,
                max_new_tokens=params.max_new_tokens,
            )

        # 6) Decode & strip (identical behavior)
        out_text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        if conv.sep_style == SeparatorStyle.TWO:
            out_text = out_text.split(conv.sep2)[-1].strip()
        elif conv.sep_style == SeparatorStyle.LLAMA_2:
            out_text = out_text.split("[/INST]")[-1].strip()

        return out_text
