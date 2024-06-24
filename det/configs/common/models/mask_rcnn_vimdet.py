from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.modeling import VisionMambaDet, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .mask_rcnn_fpn import model
from ..data.constants import constants

model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"

# Base
embed_dim, depth, num_heads, dp = 768, 24, 24, 0.1
# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(VisionMambaDet)(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        drop_path_rate=dp,
        out_feature="last_feat",
        last_layer_process="add",
        bimamba_type="v2",
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_abs_pos_embed=True,
        if_rope=True,
        if_rope_residual=True,
        pt_hw_seq_len=14,
        if_cls_token=False,
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)

model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "LN"

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]
