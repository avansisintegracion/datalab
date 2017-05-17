---
title: Partage de connaissance
subtitle: Compétition kaggle 🐟
author: Cristian, Mikael
date: 2017-05-23
---

# About the competition

## Starting point

In the Western and Central Pacific, 60% of the world’s tuna is caught illegally, a threat to marine ecosystem.

## Goal of the competition

Automate fish detection on pictures from fishing boats.
(with machine learning)

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

## Preliminary observations

* Pictures from video sequences
* Limited number of boats in training set
* Day/night pictures
* Multiple fishes per picture
* Train set labelling errors

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

# A tea break, working with two minds
(a note on methods)

## Cookiecutter

```shell
├── LICENSE
├── Makefile           <- Makefile with commands like 'make data' or 'make train'
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator s initials, and a short '-' delimited description, e.g.
│                         '1.0-jqp-initial-data-exploration'.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with 'pip freeze > requirements.txt'
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

```


## Data abstraction layer

Every picture was

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
