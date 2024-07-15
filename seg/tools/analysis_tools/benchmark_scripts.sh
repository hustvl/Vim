# ViM benchmark

cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/work_dirs/adj2/best_mIoU_iter_55000.pth 

cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/work_dirs/adj2/best_mIoU_iter_55000.pth 

# DeiT benchmark

cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/work_dirs/upernet_deit_tiny_12_512_slide_60k/upernet_deit_tiny_12_512_slide_60k.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/work_dirs/upernet_deit_tiny_12_512_slide_60k/best_mIoU_iter_58000.pth 

cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/deit/upernet/upernet_deit_tiny_12_512_slide_60k.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/work_dirs/upernet_deit_tiny_12_512_slide_60k/best_mIoU_iter_58000.pth 

# UperNet deit
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/deit/upernet/upernet_deit_tiny_12_512_slide_60k.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/work_dirs/upernet_deit_tiny_12_512_slide_60k/best_mIoU_iter_58000.pth --if-benchmark-model --input-img-size 738 --batch-size 64

# UperNet deit w/o weight
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/deit/upernet/upernet_deit_tiny_12_512_slide_60k.py --if-benchmark-model --input-img-size 512 --batch-size 64

# Backbone deit w/o weight 
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/deit/upernet/upernet_deit_tiny_12_512_slide_60k_only_backbone.py --if-benchmark-model --input-img-size 512 --batch-size 64

cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/deit/upernet/upernet_deit_tiny_12_512_slide_60k_only_backbone.py --if-benchmark-model --input-img-size 512 --batch-size 64

# Backbone deit w/o weight on zj V100
python tools/analysis_tools/benchmark.py configs/deit/upernet/upernet_deit_tiny_12_512_slide_60k_only_backbone.py --if-benchmark-model --input-img-size 512 --batch-size 64


# Backbone flash-deit w/o weight 
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/deit/upernet/upernet_flashdeit_tiny_12_512_slide_60k_only_backbone.py --if-benchmark-model --input-img-size 512 --batch-size 64

cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/deit/upernet/upernet_flashdeit_tiny_12_512_slide_60k_only_backbone.py --if-benchmark-model --input-img-size 512 --batch-size 64

# UperNet deit w/o weight + new env
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /home/zhulianghui/.conda/envs/t112-yx/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/deit/upernet/upernet_deit_tiny_12_512_slide_60k.py --if-benchmark-model --input-img-size 512 --batch-size 64

# UperNet deit no fpn
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/deit/upernet/upernet_deit_tiny_12_512_slide_60k_no_fpn.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/work_dirs/upernet_deit_tiny_12_512_slide_60k/best_mIoU_iter_58000.pth --if-benchmark-model --input-img-size 738 --batch-size 64

# UperNet vim
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/work_dirs/adj2/best_mIoU_iter_55000.pth  --if-benchmark-model --input-img-size 738 --batch-size 64

# UperNet vim w/o weight
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k.py --if-benchmark-model --input-img-size 512 --batch-size 64

# Backbone vim w/o weight 
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_only_backbone.py --if-benchmark-model --input-img-size 512 --batch-size 64

cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_only_backbone.py --if-benchmark-model --input-img-size 512 --batch-size 64

# Backbone vim w/o weight on zj V100
python tools/analysis_tools/benchmark.py configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_only_backbone.py --if-benchmark-model --input-img-size 512 --batch-size 64


# UperNet vim w/o weight + new env
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /home/zhulianghui/.conda/envs/t112-yx/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k.py --if-benchmark-model --input-img-size 512 --batch-size 64

# UperNet vim no fpn
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_no_fpn.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/work_dirs/adj2/best_mIoU_iter_55000.pth  --if-benchmark-model --input-img-size 512 --batch-size 64

# plain seg deit
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/deit/plainseg/plainseg_deit_tiny_12_512_slide_60k.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/work_dirs/plainseg_deit_tiny_12_512_slide_60k/best_mIoU_iter_59000.pth --if-benchmark-model --input-img-size 512 --batch-size 1

# plain seg vim
cd /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg ; /usr/bin/env /share/project/yxf/conda/t112/bin/python /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/tools/analysis_tools/benchmark.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/configs/vim/plainseg/plainseg_vim_tiny_24_512_slide_60k.py /share/project/lianghuizhu/JudgeLM-Project/old_ckpts/VisionProjects/seg/work_dirs/plainseg_vim_tiny_24_512_slide_60k/best_mIoU_iter_56000.pth --if-benchmark-model --input-img-size 512 --batch-size 1