## [A Framework for Real-time Object Detection and Image Restoration](https://arxiv.org/abs/2303.09190)
### Overall two-stage framework
<p align="center">
  <img src="img/figure_stage.jpg" width="640" title="Stage-1">
</p>

### SwinOIR Network Architecture
<p align="center">
  <img src="img/figure_swinoir.jpg" width="640" title="Stage-2">
</p>

## Abstract
Object detection and single image super-resolution are classic problems in computer vision (CV). The object detection task aims to recognize the objects in input images, while the image restoration task aims to reconstruct high quality images from given low quality images. In this paper, a two-stage framework for object detection and image restoration is proposed. The first stage uses YOLO series algorithms to complete the object detection and then performs image cropping. In the second stage, this work improves Swin Transformer and uses the new proposed algorithm to connect the Swin Transformer layer to design a new neural network architecture. We name the newly proposed network for image restoration SwinOIR. This work compares the model performance of different versions of YOLO detection algorithms on MS COCO dataset and Pascal VOC dataset, demonstrating the suitability of different YOLO network models for the first stage of the framework in different scenarios. For image super-resolution task, it compares the model performance of using different methods of connecting Swin Transformer layers and design different sizes of SwinOIR for use in different life scenarios.


## Open-Source
For research project agreement, we don't release training code, please refer to [EDSR framework](https://github.com/sanghyun-son/EDSR-PyTorch) and our paper for details.
- [x] Paper of our method[arXiv](https://arxiv.org/abs/2303.09190)
- [x] The pretrained model and test code.

## Citation
If you find our paper useful in your research, please consider citing:

    @article{ju2023framework,
      title={A Framework for Real-time Object Detection and Image Restoration},
      author={Ju, Rui-Yang and Chen, Chih-Chia and Chiang, Jen-Shiun and Lin, Yu-Shian and Chen, Wei-Han},
      journal={arXiv preprint arXiv:2303.09190},
      year={2023}
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
### Example of experimental results of two-stage framework on BSDS300 Dataset
<p align="center">
  <img src="img/figure_result_bsds.jpg" width="640" title="Stage-1">
</p>

## Quantitative comparison (average PSNR/SSIM) of image SR on benchmark datasets using different methods
| Connection Method | Set5-PSNR↑ | Set14-PSNR↑ | BSD100-PSNR↑ | Urban100-PSNR↑ |
| :---: | :---: | :---: | :---: | :---: |
| DSTB-5 | 30.699 | 27.677 | 26.976 | 24.552 |
| DSTB-7 | 30.692 | 27.678 | 26.983 | 24.560 |
| IDSTB-5 | **30.778** | **27.738** | **27.071** | **24.640** |
| IDSTB-7 | 30.693 | 27.662  | 26.974 | 24.539 |

## References
* [RLFN](https://github.com/bytedance/RLFN)
* [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)
* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* [SwinIR](https://github.com/JingyunLiang/SwinIR)
