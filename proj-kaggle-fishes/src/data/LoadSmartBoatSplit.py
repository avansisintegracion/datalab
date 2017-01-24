from __future__ import print_function
import os
import time
import itertools
import pickle
from glob import glob
import numpy as np
import pandas as pd

'''Feature extraction and image preprocessing'''

basedir = os.path.join('..', '..', 'raw', 'data', 'train')

classes = ['ALB',
           'BET',
           'DOL',
           'LAG',
           'SHARK',
           'YFT',
           'OTHER',
           'NoF'
           ]


def features_for(im):
    '''Whatever img preprocessing is required'''
    return array


def images():
    '''Iterate over all (image,label) pairs'''
    for ci, cl in enumerate(classes):
        images = glob('{}/{}/*.jpg'.format(basedir, cl))
        for im in sorted(images):
            yield im, ci


print('Get lists of features...')
features = []
labels = []
filenames = []
for im, ell in images():
    features.append(features_for(im))
    labels.append(ell)
    filenames.append(im)

features = np.array(features)
labels = np.array(labels)

# Load data with the split taking into account boat_group
df_80 = pickle.load(open('../../data/processed/df_80.txt', 'rb'))
# df_20 = pickle.load(open('../../data/processed/df_20.txt', 'rb'))

print('Subsample training images into sub-train and sub-test')
X_train = []
y_train = []
X_val = []
y_val = []
for row in range(0, len(features)):
    if any(df_80['img_file'] == filenames[row]):
        X_train.append(features[row])
        y_train.append(labels[row])
    else:
        X_val.append(features[row])
        y_val.append(labels[row])

X_train = np.array(X_train)
X_test = np.array(X_test)
