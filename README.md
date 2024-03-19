# Video-Portrait-Segmentation

This repository is <b>CS 6140 Machine Learning</b> group project

## Author

- Chengtian Xu
- Haoming Chen

## Dataset Download

`Video-Portrait-Segmentation` using [Tiktok Dance](https://www.kaggle.com/datasets/yasaminjafarian/tiktokdataset) as dataset.

Please download and unzip dataset before trianing.

### Dataset intruduction

This is the dataset published in CVPR 2021 introduced in the paper: [Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos](https://openaccess.thecvf.com/content/CVPR2021/html/Jafarian_Learning_High_Fidelity_Depths_of_Dressed_Humans_by_Watching_Social_CVPR_2021_paper.html)

Contains 340 videos for training and 26 videos for testing, total size is larger than 100GB. For frames in testset the size is 604\*1080, and in testset the size is 1080\*1920 and 30 fps.


## train the model

Before training, you must install python and required libraries listed in [requirements.txt](requirements.txt)

Recommend using `anaconda` to control python version.

### 1. clone this repository.

### 2. download dataset

### 3. run preprocess program.

Switch to this repository and run [dataset_prepare.py](dataset_prepare.py) first to get the path of all test files.
```
python dataset_prepare.py - p DATASET_PATH
```
Change `DATASET_PATH` to the path of `archieve` directory in the dataset directory.

If you see trian.txt file appear in dataset dirctory and no error in the console, it means that this program run successfully.

### 4. run train program.

Run [train.py](train.py) and provide the parameters you want.