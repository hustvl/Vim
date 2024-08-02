# VimSeg

## Envs.

- Python 3.9.19

  - `conda create -n your_env_name python=3.9.19`

  ```
  pip3 install numpy==1.23.4
  # install torch
  pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

  # install mmseg components
  pip install -U openmim
  mim install mmcv-full==1.7.1

  pip3 install mmsegmentation==0.29.1

  # set datasets
  change `data_root` in `seg/configs/_base_/datasets/ade20k.py`
  ```

- Meet some problems of env? Requirements for reference: seg-requirements.txt


## Train Your VimSeg

`bash scripts/ft_vim_tiny_upernet.sh`

## Test Your VimSeg

`bash scripts/eval_vim_tiny_upernet.sh`

---

## Acknowledgement
Vim semantic segmentation is built with [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2), [EVA-02](https://github.com/baaivision/EVA/tree/master/EVA-02), and [BEiT](https://github.com/microsoft/unilm/tree/master/beit/semantic_segmentation).
