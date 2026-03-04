"""
Adapted for SigLIP/SigLIP2 towers (384/14 or 512/16).
- Reads HF vision_config (handles google/siglip2-so400m-patch16-512).
- Image processor derives size from config (defaults to 512).
- No hard-coded 729 tokens; computes num patches dynamically.
"""

from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import os

from transformers.image_processing_utils import BatchFeature, get_size_dict
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
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput

from llava.utils import rank0_print


# ---------------------------
# Image Processor (dynamic)
# ---------------------------
class SigLipImageProcessor:
    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(512, 512),  # default to 512; will be overridden by tower config
        crop_size: Dict[str, int] = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
    ):
        if crop_size is None:
            crop_size = {"height": size[0], "width": size[1]}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        # Accept a single PIL image or a list/iterable of images
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # normalize to numpy arrays / PIL
            images = [to_numpy_array(img) if not isinstance(img, Image.Image) else img for img in images]

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)


# ---------------------------
# Vision Config (reads HF)
# ---------------------------
class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=512,           # default to 512; HF config will override
        patch_size=16,            # default to 16; HF config will override
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # SigLIP repos may store the vision settings nested under "vision_config"
        mt = config_dict.get("model_type")
        if (mt in {"siglip", "siglip2", "siglip_text_vision_model"} or "vision_config" in config_dict) and isinstance(
            config_dict.get("vision_config"), dict
        ):
            config_dict = config_dict["vision_config"]

        # fall back to direct dict
        return cls.from_dict(config_dict, **kwargs)


# ---------------------------
# Outputs
# ---------------------------
@dataclass
class SigLipVisionModelOutput(ModelOutput):
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# ---------------------------
# Embeddings
# ---------------------------
class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, E, H/ps, W/ps] -> [B, N, E]
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


# ---------------------------
# Attention / MLP / Layers
# ---------------------------
class SigLipAttention(nn.Module):
    """Multi-headed attention (CLIP-style)"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, num_heads={self.num_heads})"
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (bsz, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be {(bsz, self.num_heads, q_len, k_v_seq_len)}, got {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be {(bsz, 1, q_len, k_v_seq_len)}, got {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, got {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None)


class SigLipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, (attn_weights if output_attentions else None)


# ---------------------------
# Pretrained base & Encoder
# ---------------------------
class SigLipPreTrainedModel(PreTrainedModel):
    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        pass


class SigLipEncoder(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False  # can be toggled externally if needed

    def _ckpt(self, func, *args):
        return torch.utils.checkpoint.checkpoint(lambda *i: func(*i), *args)

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states, attn = self._ckpt(layer, hidden_states, attention_mask, output_attentions)
            else:
                hidden_states, attn = layer(hidden_states, attention_mask, output_attentions=output_attentions)
            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


# ---------------------------
# Vision Transformer & Head
# ---------------------------
class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state):
        bsz = hidden_state.shape[0]
        probe = self.probe.repeat(bsz, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)

        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)
        self.vision_model = SigLipVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# ---------------------------
# Vision Tower wrapper
# ---------------------------
class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower

        # Temporary defaults; will be overwritten after load
        self.config = SigLipVisionConfig()
        self.image_processor = SigLipImageProcessor(size=(self.config.image_size, self.config.image_size))

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False) or (
            hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts
        ):
            rank0_print("Vision tower weights expected; loading immediately.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print(f"{self.vision_tower_name} already loaded; skipping.")
            return

        # Read HF config (handles 512/16 vs 384/14)
        cfg = SigLipVisionConfig.from_pretrained(self.vision_tower_name)

        # Build model with matching config; shapes will align with checkpoint
        self.vision_tower = SigLipVisionModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
            config=cfg,
            # If you intentionally mix sizes, uncomment the next line:
            # ignore_mismatched_sizes=True,
        )

        # Sync tower/processor to loaded config
        self.config = self.vision_tower.config
        self.image_processor = SigLipImageProcessor(size=(self.config.image_size, self.config.image_size))

        # LLaVA-style: remove last layer & pooling head (we output patch features)
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

        # Helpful print once
        rank0_print(
            f"[SigLIP] image_size={self.config.image_size} "
            f"patch_size={self.config.patch_size} hidden={self.config.hidden_size} "
            f"layers={self.config.num_hidden_layers} heads={self.config.num_attention_heads}"
        )

        self.is_loaded = True

    def forward(self, images):
        # Expected sequence length (e.g., 384/14 -> 27*27=729; 512/16 -> 32*32=1024)
        expected_seq = (self.config.image_size // self.config.patch_size) ** 2

        if isinstance(images, list):
            feats = []
            for image in images:
                out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                feat = out.hidden_states[-1].to(image.dtype)
                assert feat.shape[-2] == expected_seq, f"Vision tokens {feat.shape[-2]} != expected {expected_seq}"
                feats.append(feat)
            image_features = feats
        else:
            outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = outs.hidden_states[-1].to(images.dtype)
            assert image_features.shape[-2] == expected_seq, f"Vision tokens {image_features.shape[-2]} != expected {expected_seq}"

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
