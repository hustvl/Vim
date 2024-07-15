#!/bin/bash
# bash /client-tools/repair_A100.sh
source /mnt/bn/lianghuidata/miniconda/bin/activate /mnt/bn/lianghuidata/miniconda/envs/vim-seg
cd /mnt/bn/lianghuidata/Vim/seg

SEG_CONFIG=configs/vim/upernet/upernet_vim_small_24_512_slide_60k.py
TRAINED_CKPT=/mnt/bn/lianghuidata/ckpts/vim/seg/vim-s-upernet-iter-60000.pth

python test.py ${SEG_CONFIG} ${TRAINED_CKPT} --eval mIoU \
    --options model.backbone.if_bimamba=False model.backbone.bimamba_type=v2 optimizer.lr=1e-4 model.backbone.use_residual_as_feature=True model.backbone.last_layer_process=add optimizer.paramwise_cfg.layer_decay_rate=0.95