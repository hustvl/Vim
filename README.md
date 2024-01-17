<div align="center">
<h1>Vision Mamba </h1>
<h3>Efficient Visual Representation Learning with Bidirectional State Space Model</h3>

[Lianghui Zhu](https://github.com/Unrealluver)<sup>1</sup> \*,[Bencheng Liao](https://github.com/LegendBC)<sup>1</sup> \*,[Qian Zhang](https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN)<sup>2</sup>, [Xinlong Wang](https://www.xloong.wang/)<sup>3</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Xinggang Wang](https://xinggangw.info/)<sup>1 :email:</sup>

<sup>1</sup>  Huazhong University of Science and Technology, <sup>2</sup>  Horizon Robotics,  <sup>3</sup> Beijing Academy of Artificial Intelligence

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.


</div>


#



### News


* **` Jan. 18th, 2024`:** We released our paper on Arxiv. Code/Models are coming soon. Please stay tuned! ☕️


## Abstract
Recently the state space models (SSMs) with efficient hardware-aware designs, i.e., Mamba, have shown great potential for long sequence modeling. Building efficient and generic vision backbones purely upon SSMs is an appealing direction. However, representing visual data is challenging for SSMs due to the position-sensitivity of visual data and the requirement of global context for visual understanding. In this paper, we show that the reliance of visual representation learning on self-attention is not necessary and propose a new generic vision backbone with bidirectional Mamba blocks (Vim), which marks the image sequences with position embeddings and compresses the visual representation with bidirectional state space models. On ImageNet classification, COCO object detection, and ADE20k semantic segmentation tasks, Vim achieves higher performance compared to well-established vision transformers like DeiT, while also demonstrating significantly improved computation & memory efficiency. For example, Vim is 2.8$\times$ faster than DeiT and saves 86.8\% GPU memory when performing batch inference on images with a resolution of 1248$\times$1248. The results demonstrate that Vim is capable of overcoming the computation & memory constraints on performing Transformer-style understanding for high-resolution images and it has great potential to become the next-generation backbone for vision foundation models.


<div align="center">
<img src="assets/vim_teaser_v1.7.png" />
</div>

