## Object recognition and computer vision 2018/2019

### Assignment 3: Image classification

This is an implementation that achieves 91.712 % in the Kaggle challenge MVA 2018-2019

https://www.kaggle.com/c/mva-recvis-2018

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18/assignment3/bird_dataset.zip). The test image labels are not provided.

#### Training and validating your model

##### Crop the image

run the jupyter crop_bird

##### Extract the features

git clone the repo https://github.com/richardaecn/cvpr18-inaturalist-transfer

Run it for the global images and the cropped images

##### Evaluate

Run  main&evaluation_Regression.py

#### Acknowledgments

Adapted from https://github.com/richardaecn/cvpr18-inaturalist-transfer

@inproceedings{Cui2018iNatTransfer,
  title = {Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning},
  author = {Yin Cui, Yang Song, Chen Sun, Andrew Howard, Serge Belongie},
  booktitle={CVPR},
  year={2018}
}

and https://github.com/chainer/chainercv

@inproceedings{ChainerCV2017,
    author = {Niitani, Yusuke and Ogawa, Toru and Saito, Shunta and Saito, Masaki},
    title = {ChainerCV: a Library for Deep Learning in Computer Vision},
    booktitle = {ACM Multimedia},
    year = {2017},
}