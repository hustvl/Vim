from functools import partial

from .cascade_mask_rcnn_vitdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.init_checkpoint = "/share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/deit/output/a-deit-weights/deit_small_patch16_224-cd65a155.pth"

model.backbone.net.embed_dim = 384
model.backbone.net.depth = 12
model.backbone.net.num_heads = 6
model.backbone.net.use_rel_pos = False
