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

model.backbone.net.embed_dim = 384
model.backbone.net.depth = 24
model.backbone.net.if_cls_token = True
model.backbone.net.pretrained = "/share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/deit/output/a-vim-weights/bc_vim_s_79p8acc.pth"
optimizer.params.lr_factor_func = partial(get_vim_lr_decay_rate, num_layers=24, lr_decay_rate=0.837)