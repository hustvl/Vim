#!/bin/bash
# bash /client-tools/repair_A100.sh
source /mnt/bn/lianghuidata/miniconda/bin/activate /mnt/bn/lianghuidata/miniconda/envs/vim-seg
cd /mnt/bn/lianghuidata/Vim/seg

SEG_CONFIG=configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k.py
TRAINED_CKPT=/mnt/bn/lianghuidata/ckpts/vim/seg/vim-t-upernet-iter-60000.pth

python test.py ${SEG_CONFIG} ${TRAINED_CKPT} --eval mIoU