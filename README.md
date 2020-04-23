# UNet: semantic segmentation with PyTorch

Adapted for the segmentation of radar images.
Hugues Talbot November 19 2019.
Latest version: April 23, 2020.

1- Make sure you have a recent enough version of Python, pytorch, etc. Everything is specified in Pytorch-UNet/requirements.txt

we recommend you install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

## How to train

1- make sure data/input and data/masks point to the training data

2- run the following command:

```
python ./train.py --help

usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]                    
                                                                                               
Train the UNet on images and target masks                                                       
                                                                                                
optional arguments:                                                                             
  -h, --help            show this help message and exit                                         
  -e E, --epochs E      Number of epochs (default: 5)                                           
  -b [B], --batch-size [B]                                                                      
                        Batch size (default: 1)                                                 
  -l [LR], --learning-rate [LR]                                                                 
                        Learning rate (default: 0.1)                                            
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)                            
  -s SCALE, --scale SCALE                                                                       
                        Downscaling factor of the images (default: 1.0)                         
  -v VAL, --validation VAL                                                                      
                        Percent of the data that is used as validation (0-100)                  
                        (default: 15.0)                                     
```

3- we recommend:

`python ./train.py -e 100 -b 1`

i.e: 100 epoch and a batch-size of 1. There is not enough data in the training set for large batch sizes

4- The training works at about 30 images per second. There are 1066 images in the training set (85% of the total number of labeled images), so it takes about 35s per epoch. 100 epochs therefore take about 30mn

5- On the present training data, the expected final Dice should be around 0.65.



## How to test

- select a good trained model (see below)
- select a series of input data, e.g. in the example below `data/input/Video_20190327_ant7_???.png`
- run the following command from the top directory (containing predict.py)

`time python ./predict.py  --model models/MODEL_dice_0_67.pth --input data/input/Video_2019327_ant7_???.png`

- The results will be in `data/input/Video_20190327_ant7_???_OUT.png`


## In case of problems

Please report problems via github, so we can track them [here](https://github.com/hugues-talbot/Pytorch-UNet)


# This is free software (as in free speech)
Note however that the models weights produced by this software are your own, and the models are standard

## This is the original README

Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch for Kaggle's [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) from high definition images.

This model was trained from scratch with 5000 images (no data augmentation) and scored a [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of 0.988423 (511 out of 735) on over 100k test images. This score could be improved with more training, data augmentation, fine tuning, playing with CRF post-processing, and applying more weights on the edges of the masks.

The Carvana data is available on the [Kaggle website](https://www.kaggle.com/c/carvana-image-masking-challenge/data).

## Usage
**Note : Use Python 3**
### Prediction

You can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```shell script
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
```
You can specify which model file to use with `--model MODEL.pth`.

### Training

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 15.0)

```
By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively.

## Tensorboard
You can visualize in real time the train and test losses, along with the model predictions with tensorboard:

`tensorboard --logdir=runs`

## Notes on memory

The model has be trained from scratch on a GTX970M 3GB.
Predicting images of 1918*1280 takes 1.5GB of memory.
Training takes much approximately 3GB, so if you are a few MB shy of memory, consider turning off all graphical displays.
This assumes you use bilinear up-sampling, and not transposed convolution in the model.

---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)

## Note on merging with upstream

https://reflectoring.io/github-fork-and-pull/

Other developers don’t sleep while you are coding. Thus, it may happen that while you are editing your fork (step #3) other changes are made to the original repository. To fetch these changes into your fork, use these commands in your fork workspace:

# add the original repository as remote repository called "upstream"
git remote add upstream https://github.com/OWNER/REPOSITORY.git

# fetch all changes from the upstream repository
git fetch upstream

# switch to the master branch of your fork
git checkout master

# merge changes from the upstream repository into your fork
git merge upstream/master
