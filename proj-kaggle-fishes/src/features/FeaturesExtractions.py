from __future__ import print_function
import os
import os.path as op
import time
import itertools
import sys
import pickle
from glob import glob
import numpy as np
import pandas as pd
import mahotas as mh
from mahotas.features import surf
from sklearn.cluster import KMeans
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


INTERIM = '../../data/interim'


class GetFeatures(object):
    '''Feature extraction and image preprocessing'''
    def __init__(self):
        self.basedir = op.join(INTERIM, 'train', 'crop')
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
        self.keras_features = []

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
            with open(op.join(INTERIM, 'ifeatures.txt'), 'wb') as file:
                pickle.dump(self.ifeatures, file)

            with open(op.join(INTERIM, 'labels.txt'), 'wb') as file:
                pickle.dump(self.labels, file)

            with open(op.join(INTERIM, 'filenames.txt'), 'wb') as file:
                pickle.dump(self.filenames, file)

            with open(op.join(INTERIM, 'subsample.txt'), 'wb') as file:
                pickle.dump(self.subsample, file)
        except:
            print('IFeatures improperly dumped')

        return print('IFeatures extracted')

    def SURFextractor(self):
        if op.exists(op.join(INTERIM, 'alldescriptors.txt')) is False:
            print('Computing SURF descriptors...')
            for im, _, c in self.images():
                im = mh.imread(im, as_grey=True)
                # If image is too small, no features get extracted!!
                if im.size < 6000:
                    im = mh.resize_to(im, (100, 100))
                im = im.astype(np.uint8)
                # To use dense sampling, you can try the following line:
                # alldescriptors.append(surf.dense(im, spacing=16))
                self.alldescriptors.append(surf.surf(im, descriptor_only=True))
            try:
                with open(op.join(INTERIM, 'alldescriptors.txt'), 'wb') as file:
                    pickle.dump(self.alldescriptors, file)
            except:
                print('SURF descriptors not dumped')
        else:
            try:
                print('SURF descriptors alreardy computed, loading..')
                self.alldescriptors = pickle.load(open(op.join(INTERIM, 'alldescriptors.txt'), 'rb'))
            except:
                print("Issues loading alldescriptors!")
                sys.exit(1)
        print('SURF Descriptor computation complete.')

        k = 128
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
            with open(op.join(INTERIM, 'sfeatures.txt'), 'wb') as file:
                pickle.dump(self.sfeatures, file)
        except:
            print("Sfeatures dump improperly done")

    def keras_features_extraction(self):
        print('Computing whole-image KERAS features...')
        model = VGG16(weights='imagenet', include_top=False)
        for im, ell, sub in self.images():
            im = image.load_img(im, target_size=(224, 224))
            x = image.img_to_array(im)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x)
            self.keras_features.append(features)

        self.keras_features = np.array(self.keras_features)

        try:
            with open(op.join(INTERIM, 'keras_features.txt'), 'wb') as file:
                pickle.dump(self.keras_features, file)
        except:
            print('KerasFeatures improperly dumped')

        return print('KerasFeatures extracted')

    def main(self):
        # self.wholeImage()
        # self.SURFextractor()
        self.keras_features_extraction()

if __name__ == '__main__':
    GetFeatures().main()
