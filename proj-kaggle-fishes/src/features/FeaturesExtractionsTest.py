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
        self.testdir = op.join(INTERIM, 'test', 'crop')
        self.tifeatures = []
        self.tlabels = []
        self.tfilenames = []
        self.tsubsample = []
        self.tsfeatures = []
        self.talldescriptors = []
        self.tkeras_features = []
        self.tkfeatures = []

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

    def images_test(self):
        '''Iterate over all image'''
        for im in glob('{}/*.jpg'.format(self.testdir)):
            name = os.path.basename(im)
            yield im, name

    def wholeImage(self):
        print('Computing whole-image texture features...')
        for im, name in self.images_test():
            self.tifeatures.append(self.features_for(im))
            self.tfilenames.append(name)

        self.tifeatures = np.array(self.tifeatures)
        try:
            with open(op.join(INTERIM, 'tifeatures.txt'), 'wb') as file:
                pickle.dump(self.tifeatures, file)

            with open(op.join(INTERIM, 'tlabels.txt'), 'wb') as file:
                pickle.dump(self.tlabels, file)

            with open(op.join(INTERIM, 'tfilenames.txt'), 'wb') as file:
                pickle.dump(self.tfilenames, file)

            with open(op.join(INTERIM, 'tsubsample.txt'), 'wb') as file:
                pickle.dump(self.tsubsample, file)
        except:
            print('IFeatures improperly dumped')

        return print('IFeatures extracted')

    def SURFextractor(self):
        if op.exists(op.join(INTERIM, 'talldescriptors.txt')) is False:
            print('Computing SURF descriptors...')
            for im, _ in self.images_test():
                im = mh.imread(im, as_grey=True)
                # If image is too small, no features get extracted!!
                if im.size < 6000:
                    im = mh.resize_to(im, (100, 100))
                im = im.astype(np.uint8)
                # To use dense sampling, you can try the following line:
                # alldescriptors.append(surf.dense(im, spacing=16))
                self.talldescriptors.append(surf.surf(im, descriptor_only=True))
            try:
                with open(op.join(INTERIM, 'talldescriptors.txt'), 'wb') as file:
                    pickle.dump(self.talldescriptors, file)
            except:
                print('SURF descriptors not dumped')
        else:
            try:
                print('SURF descriptors alreardy computed, loading..')
                self.talldescriptors = pickle.load(open(op.join(INTERIM, 'talldescriptors.txt'), 'rb'))
            except:
                print("Issues loading alldescriptors!")
                sys.exit(1)
        print('SURF Descriptor computation complete.')

        k = 128
        km = KMeans(k)
        concatenated = np.concatenate(self.talldescriptors)
        print('Number of descriptors: {}'.format(
                len(concatenated)))
        concatenated = concatenated[::64]
        print('Clustering with K-means...')
        km.fit(concatenated)
        for d in self.talldescriptors:
            c = km.predict(d)
            self.tsfeatures.append(np.bincount(c, minlength=k))
        self.tsfeatures = np.array(self.tsfeatures, dtype=float)
        try:
            with open(op.join(INTERIM, 'tsfeatures.txt'), 'wb') as file:
                pickle.dump(self.tsfeatures, file)
        except:
            print("Sfeatures dump improperly done")

    def keras_features_extraction(self):
        print('Computing whole-image KERAS features...')
        if op.exists(op.join(INTERIM, 'tkeras_features.txt')) is False:
            model = VGG16(weights='imagenet', include_top=False)
            for im, ell, sub in self.images_test():
                im = image.load_img(im, target_size=(224, 224))
                x = image.img_to_array(im)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = model.predict(x)
                self.tkeras_features.append(features)

            self.tkeras_features = np.array(self.tkeras_features)

            try:
                with open(op.join(INTERIM, 'tkeras_features.txt'), 'wb') as file:
                    pickle.dump(self.tkeras_features, file)
            except:
                print('KerasFeatures improperly dumped')
                sys.exit(1)
        else:
            try:
                print('Keras descriptors alreardy computed, loading..')
                self.tkeras_features = pickle.load(open(op.join(INTERIM, 'tkeras_features.txt'), 'rb'))
            except:
                print("Issues loading alldescriptors!")
                sys.exit(1)

        for img in self.keras_features:
            self.tkfeatures.append(img.flatten())
        self.tkfeatures = np.array(self.tkfeatures)

        try:
            with open(op.join(INTERIM, 'tkfeatures.txt'), 'wb') as file:
                pickle.dump(self.tkfeatures, file)
        except:
            print('KerasFeatures improperly dumped')

        return print('KerasFeatures extracted')

    def main(self):
        self.wholeImage()
        self.SURFextractor()
        # self.keras_features_extraction()

if __name__ == '__main__':
    GetFeatures().main()
