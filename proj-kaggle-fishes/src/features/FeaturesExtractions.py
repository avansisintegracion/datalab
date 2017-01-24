from __future__ import print_function
import os
import time
import itertools
import pickle
from glob import glob
import numpy as np
import pandas as pd
import mahotas as mh
from mahotas.features import surf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

INTERIM = '../../data/interim'


class GetFeatures(object):
    '''Feature extraction and image preprocessing'''
    def __init__(self):
        self.basedir = os.path.join(INTERIM, 'train', 'crop')
        self.classes = ['ALB',
                        'BET',
                        'DOL',
                        'LAG',
                        'SHARK',
                        'YFT',
                        'OTHER',
                        'NoF'
                        ]
        self.ifeatures = []
        self.labels = []
        self.filenames = []
        self.subsample = []
        self.sfeatures = []
        self.alldescriptors = []

    def chist(self, im):
        '''Compute color histogram of input image

        Parameters
        ----------
        im : ndarray
            should be an RGB image

        Returns
        -------
        c : ndarray
            1-D array of histogram values
        '''

        # Downsample pixel values:
        im = im // 64

        # We can also implement the following by using np.histogramdd
        # im = im.reshape((-1,3))
        # bins = [np.arange(5), np.arange(5), np.arange(5)]
        # hist = np.histogramdd(im, bins=bins)[0]
        # hist = hist.ravel()

        # Separate RGB channels:
        r, g, b = im.transpose((2, 0, 1))

        pixels = 1 * r + 4 * g + 16 * b
        hist = np.bincount(pixels.ravel(), minlength=64)
        hist = hist.astype(float)
        return np.log1p(hist)

    def features_for(self, im):
        '''Extract color histogram from image'''
        im = mh.imread(im)
        img = mh.colors.rgb2grey(im).astype(np.uint8)
        return np.concatenate([mh.features.haralick(img).ravel(),
                               self.chist(im)])

    def images(self):
        '''Iterate over all (image,label) pairs'''
        for ci, cl in enumerate(self.classes):
            images = glob('{}/*/{}/*.jpg'.format(self.basedir, cl))
            for im in sorted(images):
                yield im, ci, str(im.split('/')[-3])

    def wholeImage(self):
        print('Computing whole-image texture features...')
        for im, ell, sub in self.images():
            self.ifeatures.append(self.features_for(im))
            self.labels.append(ell)
            self.filenames.append(im)
            self.subsample.append(sub)

        self.ifeatures = np.array(self.ifeatures)
        self.labels = np.array(self.labels)
        try:
            with open(os.path.join(INTERIM, 'ifeatures.txt'), 'wb') as file:
                pickle.dump(self.ifeatures, file)

            with open(os.path.join(INTERIM, 'labels.txt'), 'wb') as file:
                pickle.dump(self.labels, file)

            with open(os.path.join(INTERIM, 'filenames.txt'), 'wb') as file:
                pickle.dump(self.filenames, file)

            with open(os.path.join(INTERIM, 'subsample.txt'), 'wb') as file:
                pickle.dump(self.subsample, file)
        except:
            print('IFeatures improperly dumped')

        return print('IFeatures extracted')

    def SURFextractor(self):
        print('Computing SURF descriptors...')
        for im, _, c in self.images():
            im = mh.imread(im, as_grey=True)
            im = im.astype(np.uint8)
            # To use dense sampling, you can try the following line:
            # alldescriptors.append(surf.dense(im, spacing=16))
            self.alldescriptors.append(surf.surf(im, descriptor_only=True))
        try:
            with open(os.path.join(INTERIM, 'alldescriptors.txt'), 'wb') as file:
                pickle.dump(self.alldescriptors, file)
        except:
            print('SURF descriptors not dumped')
        print('SURF Descriptor computation complete.')

        k = 256
        km = KMeans(k)
        concatenated = np.concatenate(self.alldescriptors)
        print('Number of descriptors: {}'.format(
                len(concatenated)))
        concatenated = concatenated[::64]
        print('Clustering with K-means...')
        km.fit(concatenated)
        for d in self.alldescriptors:
            c = km.predict(d)
            self.sfeatures.append(np.bincount(c, minlength=k))
        self.sfeatures = np.array(self.sfeatures, dtype=float)
        try:
            with open(os.path.join(INTERIM, 'sfeatures.txt'), 'wb') as file:
                pickle.dump(self.sfeatures, file)
        except:
            print("Sfeatures dump improperly done")

    def main(self):
        self.wholeImage()
        # self.SURFextractor()


if __name__ == '__main__':
    GetFeatures().main()
