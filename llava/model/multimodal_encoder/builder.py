# llava/model/multimodal_encoder/builder.py
# import os   # <-- ADD THIS LINE
#from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
#from .imagebind import ImageBindWrapper
#from .open_clip_encoder import OpenCLIPVisionTower
#from .hf_vision import HFVisionTower
#from .siglip_encoder import SigLipVisionTower
#from .mlcd_encoder import MLCDVisionTower, MLCDVisionTowerS2

# ✅ correct import
# from .vision_tower_deepseekocr_official import DeepSeekOCRVisionTower
from .vision_tower_deepseekocr import DeepSeekOCRVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower",
                           getattr(vision_tower_cfg, "vision_tower", None))
    use_s2 = getattr(vision_tower_cfg, "s2", False)

    if "siglip" in vision_tower:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("hf:"):
        return HFVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower in ["imagebind_huge"]:
        return ImageBindWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("open_clip_hub"):
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "mlcd-vit-bigG-patch14" in vision_tower:
        return (MLCDVisionTowerS2 if use_s2 else MLCDVisionTower)(vision_tower, args=vision_tower_cfg, **kwargs)

    # ✅ our new name route
    elif "deepseek" in vision_tower.lower():
        # Find this section in your builder.py:
        # ADD THIS LINE:
        return DeepSeekOCRVisionTower(
            vision_tower,
            vision_tower_cfg=vision_tower_cfg,
            resolution_mode='auto',  # Adaptive resolution per image,
            enable_gundam=True,
            **kwargs
        )

    raise ValueError(f"Unknown vision tower: {vision_tower}")
