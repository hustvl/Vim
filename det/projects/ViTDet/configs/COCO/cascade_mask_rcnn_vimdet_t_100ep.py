from functools import partial

from .cascade_mask_rcnn_vimdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vim_lr_decay_rate,
)

train.init_checkpoint = ""

model.backbone.net.embed_dim = 192
model.backbone.net.depth = 24
model.backbone.net.pretrained = "/share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/deit/output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual/best_checkpoint.pth"