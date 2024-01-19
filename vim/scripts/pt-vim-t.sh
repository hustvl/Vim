#!/bin/bash
bash /client-tools/repair_A100.sh
source /opt/conda/bin/activate /home/zhulianghui/.conda/envs/deit-dev
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/deit;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token --batch-size 128 --num_workers 25 --data-path /share/project/lianghuizhu/datasets/IN1K --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual --no_amp > /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/deit/output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual.txt 2>&1
