---
title: Partage de connaissance
subtitle: Compétition kaggle 🐟 
author: Cristian, Mikaël
date: 2017-05-23
---

# About the competition

## Images classes

![](images/fish_classes.jpeg)

## Data

Name | Number of photos
---|---
Train | 3777
Test stage 1 | 1000
Test stage 2 | 12000

## Class distribution

![](images/Distribution.png){ width=50% }

## Image sizes

![](images/Images_sizes.png){ width=50% }

## Difficulties 💪

* Some image are part of video sequences (Very similar images)
* Test images come from different boats
* Day night pictures (different explosion)
* Multiple fishes per picture
* Some images are not correctly classified in the train set

## Important dates

Stage | Date 
---|---
Competition start | 14 Nov 2016
We start 🎉 | 13 Jan  2017
End stage 1 | 6 April 2017
End stage 2 | 13 April 2017

# First model

## Bag of features

* extract features with varying methods 
* find a way to combine these meaningful features
* feed them into a classifier.

# Deep learning

## Bounding box regression

* A kaggle participant posted in the [kaggle forum](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/25902) the coordinates of the bounding box for every fish in the pictures of the train set.
* This has been made using the labelling software [Sloth](https://github.com/cvhciKIT/sloth).
* Coordinates of the bounding box in terms of the starting point and the size of the box (`x`, `y`, `width` and `height`).

## Multiple fish per picture

* Only one bounding box per picture,
* Combination of the coordinates of the bounding box for each picture to include
the maximum number of fishes inside the picture.
* The pictures that does not contain bounding boxes are filled with empty box coordinates.

## Image preprocessing

* Keras provides a function
  [ImageDataGenerator](https://keras.io/preprocessing/image/) which can be used
as a preprocessing tool.
* It can modify or normalize the pictures with predefined treatment like
  rescale, rotation, shift, shear, flip, whitening, etc.
* The preprocessing generator can read the images directly from a directory
  path using the function `flow_from_directory`.
* The result can be used as an iterator with and infinite loop that generates
  images in batches.


## Training

* Keras also provides a method to train images by batches (`fit_generator`)
    * reduce memory utilization.
    * image preprocessing to be done in parallel of training process
* Requirement: bounding box coordinates and the Fish/NoFish label must be
   transformed as an iterator.  


---

### Train the model by batch 

* The generator that feed the training fonction by batch contains: 
    * The image generator
    * The bounding box coordinates generator 
    * The Fish/NoFish label

* itertools: [cylce](https://docs.python.org/2/library/itertools.html#itertools.cycle), [izip](https://docs.python.org/2/library/itertools.html#itertools.izip) 

```python
>>> itertools.cycle('ABCD') 
A B C D A B C D ...`
>>> itertools.izip('ABCD', 'xy') 
Ax By`
```

## Finetuning a pretrained model


# Conclusions

-------------

- Image preprocessing can significantly increase the performance of a
  classification algorithm.
- A feature descriptor represents a simplified version of an image by
  extracting useful information and throwing away extraneous information.
- Using feature description increases training speed compared with raw images.
