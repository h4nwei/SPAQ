# Additional Readme
## Manual for SPAQ dataset unzip at linux / SPAQ数据集在linux下解压的方法
### English Version
#### Background
Since SPAQ.zip is a large file and cannot be directly unzipped in linux device with “unzip”, the author compresses SPAQ into splits. But there is still the possibility of data corruption. I have recorded the problems and solutions encountered after decompressing large files. I hope it could contribute to the SPAQ or IQA community.

#### How To Unzip The Split Zip?
1. Precondition
All the *.z* file (including .zip) have been set at linux destination device.
2. Command
```
zip -FF SPAQ.zip --out SPAQ_fixed.zip
unzip SPAQ_fixed.zip
```

#### Possible Problems and Solutions
1. "OSErr" was throw during the running of the program
I found some data was damaged. These images would cause OSErr when they were about to be convert to RGB during the running of the program. It is insane to manually check which images are demaged in 11125 images, so I write two pieces of python code to check demaged images and save the undemaged images that needed to be re-uploaded. `OutputErrorImages.py` should be run at your destination device that possess demaged data. `pickImage.py` should be run at your device that possess the undemaged SPAQ dataset (normally at the device which downloads data from Baiduyunpan or GoogleDriveCould).

### 中文版
#### 背景
由于SPAQ.zip是大文件，在linux中无法直接unzip解压，所以作者将SPAQ分卷压缩。但即使这样，仍然可能出现数据受损的情况。我记录了大文件解压后遇到的问题和解决办法，希望能帮助到你。

#### 如何在linux中解压分卷呢？
1. 前提
所有*.z*文件（包括.zip）都已经被放在linux文件夹下。
2. 解压指令
```
zip -FF SPAQ.zip --out SPAQ_fixed.zip
unzip SPAQ_fixed.zip
```

#### 可能遇到的问题和解决办法
1. python读取数据过程中报错：“OSErr”
经过排查，我发现部分图像出现了图像数据部分丢失的情况，这部分图像在被 convert 成 RGB 图像时会丢出 OSErr。人工检查数据集中哪些图像有误是不实际的，所以我写了两段 python 代码分别用于检查损坏的图像和另存一份需要重新上传的未损坏的图像。你需要在linux设备上运行 `OutputErrorImages.py`，并把它的输出复制后保存到你本地设备的 `demagedImgs.txt` 文件中。接着，在你的本地设备运行 `pickImage.py`，以将未损坏的图像保存到 `updated` 文件夹中，以便于你进一步将这些图像重新上传到 linux 终端。

### Codes
For directly apply my codes, your dir tree should be like this:
![dir tree](./dirtree.jpg)

```
### OutputErrorImages.py
### This file is to Test which image is be demaged, and print their name in console. You should copy the output and paste them at your local device as "demagedImgs.txt" and run 'python pickImage.py'.

import os
import os.path
from openpyxl import load_workbook
import torchvision
from torchvision.transforms import transforms
from PIL import Image
import scipy.io
import xlrd
from tqdm import tqdm

## DEFINE Args
root = "" # your SPAQ dataset location
index = list(range(0, 11125))
transform = transforms = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

## START loading SPAQ dataset
data = xlrd.open_workbook(os.path.join(root, 'Annotations/MOS and Image attribute scores.xlsx'))
table = data.sheet_by_index(0)

for rowNum in tqdm(range(table.nrows)):
    if rowNum > 0:
        rowValue = table.row_values(rowNum)
        # acquire imgnames, convert them into RGB. If OSErr was throw, then print their name for record.
        try:
            sample = pil_loader(os.path.join(root, 'TestImage', rowValue[0]))
        except OSError as e:
            print(os.path.join(root, 'TestImage', rowValue[0]))
print("Success")
```


```
### pickImage.py 
# This file is to copy the undemaged images into dst dir, for use to re-upload these images.

import os
from PIL import Image
import os.path
import shutil

dst = "./updated/" # image folder which should contains the indamaged images correspponding to the demaged images.
file_name = "demagedImgs.txt"

def openreadtxt(file_name):
    file = open(file_name,'r')  
    file_data = file.readlines() 
    for row in file_data:
        row = row.strip('\n')
        shutil.copy(f"TestImage/{row}", dst)

openreadtxt(file_name)
print("Success")
```

```
### data transfer tool
### This command is for those whose devide could run linux command. I prefer to transfer data through scp command than using APP SFTP. Window user could run this file in "git bash"!
scp -r -P 22 ./updated.zip user@xx.xx.xx.xx:/dst_location
```




---------------------------------------------------------------------------------------------------------
# Perceptual Quality Assessment of Smartphone Photography
This repository contains the constructed Smartphone Photography Attribute and Quality (SPAQ) database and implementations for the paper "Perceptual Quality Assessment of Smartphone Photography",
 [Yuming Fang*](http://sim.jxufe.cn/JDMKL/ymfang.html), Hanwei Zhu*, Yan Zeng, [Kede Ma](https://kedema.org/), [Zhou Wang](https://ece.uwaterloo.ca/~z07wang/), *IEEE Conference on Computer Vision and Pattern Recognition*, 2020. (*Equal contribution)

Download:

&emsp;　![](./images/icon_pdf.png) &emsp;　&emsp;　&emsp;　![](./images/icon_pdf.png) &emsp;　&emsp;　&emsp;　![](./images/icon_zip.png)　&emsp;　&emsp;　![](./images/icon_pdf.png)<br>
&emsp;　&emsp;　&ensp; [paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf)　&emsp;　&emsp;　&emsp;　&emsp;　&ensp;  [supplementary](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Fang_Perceptual_Quality_Assessment_CVPR_2020_supplemental.pdf)　&emsp;　&emsp;　&emsp;　&ensp;　SPAQ database &emsp;　&emsp; &emsp;　&ensp;　[PPT](https://drive.google.com/file/d/1rhSgzAtAiryenF-xK9A8etoOlahcdGZZ/view?usp=sharing)


		

## Introduction
As smartphones become people's primary cameras to take photos, the quality of their cameras and the associated computational photography modules has become a de facto standard in evaluating and ranking smartphones in the consumer market. We conduct so far the most comprehensive study of perceptual quality assessment of smartphone photography. We introduce the Smartphone Photography Attribute and Quality (SPAQ) database, consisting of 11,125 pictures taken by 66 smartphones, where each image is attached with so far the richest annotations. Specifically, we collect a series of human opinions for each image, including image quality, image attributes (brightness, colorfulness, contrast, noisiness, and sharpness), and scene category labels (animal, cityscape, human, indoor scene, landscape, night scene, plant, still life, and others) in a well-controlled laboratory environment. The exchangeable image file format (EXIF) data for all images are also recorded to aid deeper analysis. We also make the first attempts using the database to train blind image quality assessment (BIQA) models constructed by baseline and multi-task deep neural networks. The results provide useful insights on how EXIF data, image attributes and high-level semantics interact with image quality, how next-generation BIQA models can be designed, and how better computational photography systems can be optimized on mobile devices.



## Database
The SPAQ database and the annotations (MOS, image attributes scores, EXIF tags, and scene catogory labels) can be downloaded at the [**Baidu Yun**](https://pan.baidu.com/s/18YzAtXb4cGdBGAsxuEVBOw) (Code: b29m) or [**Google drive**](https://drive.google.com/drive/u/1/folders/1wZ6HOHi5h43oxTe2yLYkFxwHPgJ9MwvT).


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
