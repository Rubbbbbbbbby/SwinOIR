# SwinOIR

> [Resolution Enhancement Processing on Low Quality Images Using Swin Transformer Based on Interval Dense Connection Strategy](https://arxiv.org/abs/2303.09190)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-framework-for-real-time-object-detection/image-super-resolution-on-bsd100-3x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-bsd100-3x-upscaling?p=a-framework-for-real-time-object-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-framework-for-real-time-object-detection/image-super-resolution-on-set14-3x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set14-3x-upscaling?p=a-framework-for-real-time-object-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-framework-for-real-time-object-detection/image-super-resolution-on-set5-2x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set5-2x-upscaling?p=a-framework-for-real-time-object-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-framework-for-real-time-object-detection/image-super-resolution-on-set5-3x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set5-3x-upscaling?p=a-framework-for-real-time-object-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-framework-for-real-time-object-detection/image-super-resolution-on-urban100-3x)](https://paperswithcode.com/sota/image-super-resolution-on-urban100-3x?p=a-framework-for-real-time-object-detection)

### SwinOIR Network Architecture
<p align="center">
  <img src="img/figure_swinoir.jpg" width="640" title="swinoir">
</p>

## Open-Source
For research project agreement, we don't release training code, please refer to [EDSR framework](https://github.com/sanghyun-son/EDSR-PyTorch) and our paper for details.
- [x] Paper of our method [[arXiv]](https://arxiv.org/abs/2303.09190)
- [x] The pretrained model and test code.

## Citation
If you find our paper useful in your research, please consider citing:

    @article{ju2023resolution,
      title={Resolution enhancement processing on low quality images using swin transformer based on interval dense connection strategy},
      author={Ju, Rui-Yang and Chen, Chih-Chia and Chiang, Jen-Shiun and Lin, Yu-Shian and Chen, Wei-Han},
      journal={Multimedia Tools and Applications},
      pages={1--17},
      year={2023},
      publisher={Springer}
    }
  
## Requirements
* Linux (Ubuntu)
* Python >= 3.6
* Pytorch >= 1.5.0
* NVIDIA GPU + CUDA CuDNN

## Dataset
* DIV2K Dataset [(Download Link)](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
* Set5 Dataset [(Download Link)](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
* Set14 Dataset [(Download Link)](https://sites.google.com/site/romanzeyde/research-interests)
* B100 Dataset [(Download Link)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
* Urban100 Dataset [(Download Link)](https://www.kaggle.com/datasets/harshraone/urban100)

## Experimental Results
<p align="center">
  <img src="img/figure_experiment.jpg" width="640" title="experiment">
</p>

## Application
### Overall two-stage framework
<p align="center">
  <img src="img/figure_stage.jpg" width="640" title="stage">
</p>

### Graphical User Interface 
<p align="center">
  <img src="img/figure_app.jpg" width="640" title="app">
</p>

## References

<details><summary> <b>GitHub</b> </summary>
  
* [RLFN](https://github.com/bytedance/RLFN)
* [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)
* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* [SwinIR](https://github.com/JingyunLiang/SwinIR)
