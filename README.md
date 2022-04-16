# Perceptual Quality Assessment of Smartphone Photography
This repository contains the constructed Smartphone Photography Attribute and Quality (SPAQ) database and implementations for the paper "Perceptual Quality Assessment of Smartphone Photography",
 [Yuming Fang*](http://sim.jxufe.cn/JDMKL/ymfang.html), Hanwei Zhu*, Yan Zeng, [Kede Ma](https://kedema.org/), [Zhou Wang](https://ece.uwaterloo.ca/~z07wang/), *IEEE Conference on Computer Vision and Pattern Recognition*, 2020. (*Equal contribution)

Download:

&emsp;　![](./images/icon_pdf.png) &emsp;　&emsp;　&emsp;　![](./images/icon_pdf.png) &emsp;　&emsp;　&emsp;　![](./images/icon_zip.png)　&emsp;　&emsp;　![](./images/icon_pdf.png)<br>
&emsp;　&emsp;　&ensp; [paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf)　&emsp;　&emsp;　&emsp;　&emsp;　&ensp;  [supplementary](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Fang_Perceptual_Quality_Assessment_CVPR_2020_supplemental.pdf)　&emsp;　&emsp;　&emsp;　&ensp;　SPAQ database &emsp;　&emsp;　&ensp;　[PPT](https://drive.google.com/file/d/1rhSgzAtAiryenF-xK9A8etoOlahcdGZZ/view?usp=sharing)


		

## Introduction
As smartphones become people's primary cameras to take photos, the quality of their cameras and the associated computational photography modules has become a de facto standard in evaluating and ranking smartphones in the consumer market. We conduct so far the most comprehensive study of perceptual quality assessment of smartphone photography. We introduce the Smartphone Photography Attribute and Quality (SPAQ) database, consisting of 11,125 pictures taken by 66 smartphones, where each image is attached with so far the richest annotations. Specifically, we collect a series of human opinions for each image, including image quality, image attributes (brightness, colorfulness, contrast, noisiness, and sharpness), and scene category labels (animal, cityscape, human, indoor scene, landscape, night scene, plant, still life, and others) in a well-controlled laboratory environment. The exchangeable image file format (EXIF) data for all images are also recorded to aid deeper analysis. We also make the first attempts using the database to train blind image quality assessment (BIQA) models constructed by baseline and multi-task deep neural networks. The results provide useful insights on how EXIF data, image attributes and high-level semantics interact with image quality, how next-generation BIQA models can be designed, and how better computational photography systems can be optimized on mobile devices.



## Database
The SPAQ database and the annotations (MOS, image attributes scores, EXIF tags, and scene catogory labels) can be downloaded at the [**Baidu Yun**](https://pan.baidu.com/s/18YzAtXb4cGdBGAsxuEVBOw) (Code: b29m) or [**Google driver**](https://drive.google.com/drive/u/1/folders/1wZ6HOHi5h43oxTe2yLYkFxwHPgJ9MwvT).


## Proposed Models 
We train a baseline (BL) to predict the quality of captured images and three variants that make use of EXIF tags (MT-E), image attributes (MT-A), and scene category labels (MT-S). We provide two images to test the blind image quality assessment (BIQA) models.

### Prerequisites
The release version of BIQA models were implemented and have been tested in Ubuntu 16.04 with
- Python = 3.5.0
- PyTorch = 1.1.1
- torchvision = 0.3.0 

### Baseline Model (BL)
The pretrained checkpoint of BL can be obtained at [BL_release.pt](https://drive.google.com/file/d/1pXjXAIItViTFs7qUBY-b11WY50mzoVM2/view). To test the BL model with the default setting:
```
python BL_demo.py
``` 
### Multi-Task Learning from EXIF Tags (MT-E)
The pretrained checkpoint of MT-E can be obtained at [MT-E_release.pt](https://drive.google.com/open?id=1eEMH2wTz2bFDkpgTFpSmzdoJRTnOparJ). To test the MT-E model with the default setting:
```
python MT-E_demo.py
```

### Multi-Task Learning from Image Attributes (MT-A)
The pretrained checkpoint of MT-A can be obtained at [MT-A_release.pt](https://drive.google.com/open?id=1j0GmSgfzkB0gYu4zLVXCrlv3b3J20eOv). To test the MT-A model with the default setting:
```
python MT-A_demo.py
```

### Multi-Task Learning from Scene Semantics (MT-S）
The pretrained checkpoint of MT-S can be obtained at [MT-S_release.pt](https://drive.google.com/open?id=16m_N1neg6aDST1OqwrQRxNd-Z70_SU4o). To test the MT-S model with the default setting:
```
python MT-S_demo.py
```

## Reference

- D. Ghadiyaram and A. C. Bovik. Massive online crowdsourced study of subjective and objective picture quality. *IEEE Transactions on Image Processing*, 25(1): 372–387, Jan. 2016.
- V. Hosu, H. Lin, T. Sziranyi, and D. Saupe. KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment. *IEEE Transactions on Image Processing*, 29: 4041-4056, Jan. 2020.
- K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In *IEEE Conference on Computer vison and Pattern Recognition*, pages 770–778, 2016.

## Citation
```bibtex
@inproceedings{fang2020cvpr,
title={Perceptual Quality Assessment of Smartphone Photography},
author={Fang, Yuming and Zhu, Hanwei and Zeng, Yan and Ma, Kede and Wang, Zhou},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
pages={3677-3686},
year={2020}
}
