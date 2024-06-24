from functools import partial
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    CascadeROIHeads,
)

from .cascade_mask_rcnn_vimdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vim_lr_decay_rate,
)

from detectron2.layers import ShapeSpec

train.init_checkpoint = ""

model.backbone.net.embed_dim = 192
model.backbone.net.depth = 24
model.backbone.net.pretrained = "/share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/deit/output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual/best_checkpoint.pth"

model.backbone.out_channels = 192
model.proposal_generator.head.in_channels = 192

model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=192, height=7, width=7),
            conv_dims=[192, 192, 192, 192],
            fc_dims=[768],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=768),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes="${...num_classes}",
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

model.roi_heads.mask_head.input_shape.channels = 192
model.roi_heads.mask_head.conv_dims = [192, 192, 192, 192, 192]