#!/bin/bash
# bash /client-tools/repair_A100.sh
source /mnt/bn/lianghuidata/miniconda/bin/activate /mnt/bn/lianghuidata/miniconda/envs/det2
cd /mnt/bn/lianghuidata/Vim/det;

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1
DET_CONFIG=projects/ViTDet/configs/COCO/$DET_CONFIG_NAME.py


/mnt/bn/lianghuidata/miniconda/envs/det2/bin/python3 tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines 1 --machine-rank 0 --dist-url "tcp://127.13.44.12:60900" \
 --config-file ${DET_CONFIG} \
 train.output_dir=/mnt/bn/lianghuidata/Vim/det/work_dirs/$DET_CONFIG_NAME-4gpu-test-env \
 dataloader.train.num_workers=128 \
 dataloader.test.num_workers=8