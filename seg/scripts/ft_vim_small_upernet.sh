#!/bin/bash
# bash /client-tools/repair_A100.sh
source /mnt/bn/lianghuidata/miniconda/bin/activate /mnt/bn/lianghuidata/miniconda/envs/vim-seg
cd /mnt/bn/lianghuidata/Vim/seg

SEG_CONFIG=configs/vim/upernet/upernet_vim_small_24_512_slide_60k.py
PRETRAIN_CKPT=/mnt/bn/lianghuidata/Vim/pretrained_ckpts/pretrained-vim-s.pth

python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=45935 \
--use_env train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --work-dir work_dirs/vimseg-s --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} model.backbone.if_bimamba=False model.backbone.bimamba_type=v2 optimizer.lr=1e-4 model.backbone.use_residual_as_feature=True model.backbone.last_layer_process=add optimizer.paramwise_cfg.layer_decay_rate=0.95