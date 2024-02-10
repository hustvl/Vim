#!/bin/bash
conda activate <your_env>
cd <path_to_Vim>/vim;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 64 --drop-path 0.05 --weight-decay 0.05 --lr 1e-3 --num_workers 25 --data-path <path_to_IN1K_dataset> --output_dir ./output/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --no_amp
