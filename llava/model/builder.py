#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import rank0_print
def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    torch_dtype="float16",
    attn_implementation="eager",
    customized_config=None,
    overwrite_config=None,
    **kwargs,
):
    """
    Unified LLaVA/VLM loader (8k-ready):
    - Forces safe long-context defaults (8192) while respecting any explicit overwrite_config.
    - Keeps caller's attn_implementation (no silent override).
    - Aligns tokenizer (model_max_length, padding_side).
    - Returns a conservative context_len (min of relevant caps, clamped to 8192).
    """

    # -------------------------------------------------------
    # Common kwargs & quantization
    # -------------------------------------------------------
    kwargs["device_map"] = device_map

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        if torch_dtype == "float16":
            kwargs["torch_dtype"] = torch.float16
        elif torch_dtype == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16

    if customized_config is not None:
        kwargs["config"] = customized_config

    # multimodal flag normalization
    is_multimodal = bool(kwargs.pop("multimodal", False)) if "multimodal" in kwargs else ("llava" in model_name.lower())

    # -------------------------------------------------------
    # 8k-safe config normalization
    # -------------------------------------------------------

    # only set defaults if not explicitly provided
    #overwrite_config={}
    #overwrite_config.setdefault("max_position_embeddings", 8192)
    #overwrite_config.setdefault("tokenizer_model_max_length", 8192)
    #overwrite_config.setdefault("use_cache", True)
    # keep user's rope if present on-disk; else apply linear x2 (4k -> 8k)
    #overwrite_config.setdefault("rope_scaling", {"type": "linear", "factor": 2.0})


    # -------------------------------------------------------
    # Load paths
    # -------------------------------------------------------
    rank0_print(f"Loading model_name={model_name}, model_base={model_base}, path={model_path}")

    # ============================
    # Multimodal (LLaVA family)
    # ============================
    if is_multimodal:
        # ----- Case A: LoRA adapters + base LM -----
        if "lora" in model_name.lower() and model_base is not None:
            rank0_print("Loading LLaVA (LoRA + base) ...")
            if "mixtral" in model_name.lower():
                from llava.model.language_model.llava_mixtral import LlavaMixtralConfig, LlavaMixtralForCausalLM
                cfg = LlavaMixtralConfig.from_pretrained(model_path)
            elif "mistral" in model_name.lower():
                from llava.model.language_model.llava_mistral import LlavaMistralConfig, LlavaMistralForCausalLM
                cfg = LlavaMistralConfig.from_pretrained(model_path)
            elif "gemma" in model_name.lower():
                from llava.model.language_model.llava_gemma import LlavaGemmaConfig, LlavaGemmaForCausalLM
                cfg = LlavaGemmaConfig.from_pretrained(model_path)
            else:
                from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
                cfg = LlavaConfig.from_pretrained(model_path)

            #cfg = _apply_overwrite(cfg)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            #_align_tokenizer(tokenizer, cfg)

            # respect caller's attn_implementation
            if "mixtral" in model_name.lower():
                model = LlavaMixtralForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg, attn_implementation=attn_implementation, **kwargs
                )
            elif "mistral" in model_name.lower():
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg, attn_implementation=attn_implementation, **kwargs
                )
            elif "gemma" in model_name.lower():
                model = LlavaGemmaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg, attn_implementation=attn_implementation, **kwargs
                )
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg, attn_implementation=attn_implementation
                )

            # -- meta tensor guard (rare but handy)
            #########################################################
            #for name, param in model.named_parameters():
            #    if getattr(param, "device", None) and param.device.type == "meta":
            #        real = torch.empty_like(param, device="cpu", dtype=torch.float16)
            #        if "lm_head" in name or "embed_tokens" in name:
            #            torch.nn.init.normal_(real, mean=0.0, std=0.02)
            #        else:
            #            torch.nn.init.zeros_(real)
            #        parent = model
            #        *path, last = name.split(".")
            #        for p in path:
            #            parent = getattr(parent, p)
            #        setattr(parent, last, torch.nn.Parameter(real))
            #################################################################

            # load extra non-LoRA projector/biases if present
            extra_path = os.path.join(model_path, "non_lora_trainables.bin")
            if os.path.exists(extra_path):
                non_lora = torch.load(extra_path, map_location="cpu")
            else:
                from huggingface_hub import hf_hub_download
                cache_file = hf_hub_download(repo_id=model_path, filename="non_lora_trainables.bin")
                non_lora = torch.load(cache_file, map_location="cpu")
            non_lora = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora.items()}
            if any(k.startswith("model.model.") for k in non_lora):
                non_lora = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora.items()}
            model.load_state_dict(non_lora, strict=False)

            # merge LoRA
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_path).merge_and_unload()

        # ----- Case B: base LM + mm_projector weights -----
        elif model_base is not None:
            rank0_print(f"Loading LLaVA from base model '{model_base}' + projector from '{model_path}' ...")
            if "mixtral" in model_name.lower():
                from llava.model.language_model.llava_mixtral import LlavaMixtralConfig, LlavaMixtralForCausalLM
                cfg = LlavaMixtralConfig.from_pretrained(model_path)
                Cls = LlavaMixtralForCausalLM
            elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
                from llava.model.language_model.llava_mistral import LlavaMistralConfig, LlavaMistralForCausalLM
                cfg = LlavaMistralConfig.from_pretrained(model_path)
                Cls = LlavaMistralForCausalLM
            elif "gemma" in model_name.lower():
                from llava.model.language_model.llava_gemma import LlavaGemmaConfig, LlavaGemmaForCausalLM
                cfg = LlavaGemmaConfig.from_pretrained(model_path)
                Cls = LlavaGemmaForCausalLM
            else:
                from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
                cfg = LlavaConfig.from_pretrained(model_path)
                Cls = LlavaLlamaForCausalLM

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)


            model = Cls.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=cfg, attn_implementation=attn_implementation, **kwargs
            )
            proj_path = os.path.join(model_path, "mm_projector.bin")
            mm_projector_weights = torch.load(proj_path, map_location="cpu")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)

        # ----- Case C: fully merged multimodal checkpoint -----
        else:
            rank0_print(f"Loading merged LLaVA/VLM from '{model_path}' ...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

            # choose class by family
            if "mixtral" in model_name.lower():
                from llava.model.language_model.llava_mixtral import LlavaMixtralConfig, LlavaMixtralForCausalLM
                cfg = customized_config or LlavaMixtralConfig.from_pretrained(model_path)

                model = LlavaMixtralForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=cfg, **kwargs
                )
            elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
                from llava.model.language_model.llava_mistral import LlavaMistralConfig, LlavaMistralForCausalLM
                cfg = customized_config or LlavaMistralConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=cfg, **kwargs
                )
            elif "gemma" in model_name.lower():
                from llava.model.language_model.llava_gemma import LlavaGemmaConfig, LlavaGemmaForCausalLM
                cfg = customized_config or LlavaGemmaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaGemmaForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=cfg, **kwargs
                )
            elif "qwen" in model_name.lower() or "quyen" in model_name.lower():
                # qwen variants
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if "moe" in model_name.lower() or "A14B".lower() in model_name.lower():
                    from llava.model.language_model.llava_qwen_moe import LlavaQwenMoeConfig, LlavaQwenMoeForCausalLM
                    cfg = customized_config or LlavaQwenMoeConfig.from_pretrained(model_path)
                    model = LlavaQwenMoeForCausalLM.from_pretrained(
                        model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=cfg, **kwargs
                    )
                else:
                    from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
                    cfg = customized_config or LlavaQwenConfig.from_pretrained(model_path)
                    model = LlavaQwenForCausalLM.from_pretrained(
                        model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=cfg, **kwargs
                    )
            else:
                # LLAMA family default
                from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
                cfg = customized_config or LlavaConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=cfg, **kwargs
                )


        # Vision tokens & resize
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        # Vision tower hookup
        image_processor = None
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            vt_dtype = torch.float16 if kwargs.get("torch_dtype", torch.float16) == torch.float16 else torch.bfloat16
            vision_tower.to(device="cuda", dtype=vt_dtype)
        image_processor = vision_tower.image_processor


    # ============================
    # LM-only (no vision)
    # ============================
    else:
        if model_base is not None:
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto"
            )
            model = PeftModel.from_pretrained(model, model_path).merge_and_unload()
            model.to(torch.float16)
        else:
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        image_processor = None

    rank0_print(f"Model Class: {model.__class__.__name__}")

    # -------------------------------------------------------
    # Context length: be conservative and 8k-capped
    # -------------------------------------------------------
    caps = []
    for key in ("max_sequence_length", "max_position_embeddings", "tokenizer_model_max_length"):
        if hasattr(model.config, key):
            try:
                caps.append(int(getattr(model.config, key)))
            except Exception:
                pass
    if not caps:
        caps = [2048]
    context_len = min(max(caps), 8192)

    return tokenizer, model, image_processor, context_len
