from __future__ import print_function
import json
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
from skimage.feature import hog
from skimage import io
from mahotas.features import surf
from sklearn.cluster import KMeans
# from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import preprocess_input
from skimage.feature import blob_log
from math import sqrt
from skimage.color import rgb2gray
from skimage import io
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage import segmentation, color
from skimage.future import graph
import cv2

from src.data import DataModel


INTERIM = '../../data/interim'


class GetFeatures(object):
    '''Feature extraction and image preprocessing'''
    def __init__(self,
                 basedir=op.join(INTERIM, 'train', 'crop'),
                 classes=['ALB',
                          'BET',
                          'DOL',
                          'LAG',
                          'NoF',
                          'OTHER',
                          'SHARK',
                          'YFT'],
                 subfol=['train', 'val']):
        self.images = []
        self.f = DataModel.ProjFolder()
        self.basedir = basedir
        self.testdir = op.join(INTERIM, 'train', 'crop', 'test_stg1')
        self.classes = classes
        self.ifeatures = dict()
        self.sfeatures = dict()
        self.alldescriptors = dict()
        self.keras_features = []
        self.kfeatures = []
        self.subfol = subfol
        self.hogdescriptors = []
        self.bloblogdescriptors = []
        self.otsudescriptors = []

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

    def load_images(self):
        '''Get all images'''
        try:
            config = op.join(self.f.data_processed, 'training_images.json')
            self.images = json.load(open(config, 'rb'))
        except:
            print('Could not load training images json structure')
            print('Please run DataModel first')
            print(op.join(self.f.data_processed, 'training_images.json'))
            sys.exit(1)

        for id, im in self.images.iteritems():
            yield im['imgcrop'], id

    def wholeImage(self):
        print('Computing whole-image texture features...')
        for im, id in self.load_images():
            self.ifeatures[id] = np.array(self.features_for(im))
        try:
            with open(op.join(self.basedir, 'ifeatures.txt'), 'wb') as file:
                pickle.dump(self.ifeatures, file)
        except:
            print('IFeatures improperly dumped')

        return print('IFeatures extracted')

    def SURFextractor(self):
        if op.exists(op.join(self.basedir, 'alldescriptors.txt')) is False:
            print('Computing SURF descriptors...')
            for im, id in self.load_images():
                # im = mh.imread(im, as_grey=True)
                img = cv2.imread(im)
                # Color segmentation start here
                Z = img.reshape((-1, 3))
                # convert to np.float32
                Z = np.float32(Z)
                # define criteria, number of clusters(K) and apply kmeans()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 16
                ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                # Now convert back into uint8, and make original image
                center = np.uint8(center)
                res = center[label.flatten()]
                res2 = res.reshape((img.shape))

                b, g, r = cv2.split(res2)
                rgb_img = cv2.merge([r, g, b])
                # End of color segmentation in rgb format
                # If image is too small, no features get extracted!!
                if rgb_img.size < 6000:
                    rgb_img = mh.resize_to(rgb_img, (100, 100))
                im = rgb2gray(rgb_img).astype(np.uint8)
                # To use dense sampling, you can try the following line:
                # alldescriptors.append(surf.dense(im, spacing=16))
                self.alldescriptors[id] = surf.surf(im, descriptor_only=True)
            try:
                with open(op.join(self.basedir, 'alldescriptors.txt'), 'wb') as file:
                    pickle.dump(self.alldescriptors, file)
            except:
                print('SURF descriptors not dumped')
        else:
            try:
                print('SURF descriptors alreardy computed, loading..')
                self.alldescriptors = pickle.load(open(op.join(self.basedir, 'alldescriptors.txt'), 'rb'))
            except:
                print("Issues loading alldescriptors!")
                sys.exit(1)
        print('SURF Descriptor computation complete.')

        k = 128
        km = KMeans(k)
        concatenated = np.concatenate(list(self.alldescriptors.values()))
        print('Number of descriptors: {}'.format(
              len(concatenated)))
        concatenated = concatenated[::64]
        print('Clustering with K-means...')
        km.fit(concatenated)
        for id, d in self.alldescriptors:
            c = km.predict(d)
            self.sfeatures[id] = np.bincount(c, minlength=k)
        # self.sfeatures = np.array(self.sfeatures, dtype=float)
        try:
            with open(op.join(self.basedir, 'sfeatures.txt'), 'wb') as file:
                pickle.dump(self.sfeatures, file)
        except:
            print("Sfeatures dump improperly done")

    def keras_features_extraction(self):
        print('Computing whole-image KERAS features...')
        if op.exists(op.join(self.basedir, 'kfeatures.txt')) is False:
            model = InceptionV3(weights='imagenet', include_top=False)
            for im, id in self.load_images():
                # im = image.load_img(im, target_size=(224, 224))
                im = io.imread(im)
                # gray_resized = resize(rgb2gray(cleaned), (299, 299))
                resized = resize(im, (299, 299))
                x = image.img_to_array(resized)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = model.predict(x)
                self.keras_features[id] = features.flatten()

            try:
                with open(op.join(self.basedir, 'kfeatures.txt'), 'wb') as file:
                    pickle.dump(self.keras_features, file)
            except:
                print('KerasFeatures improperly dumped')
                sys.exit(1)
        else:
            try:
                print('Keras features alreardy computed, loading..')
                self.keras_features = pickle.load(open(op.join(self.basedir, 'kfeatures.txt'), 'rb'))
            except:
                print("Issues loading Keras Features!")
                sys.exit(1)

    def otsu_keras_features_extraction(self):
        print('Computing whole-image KERAS features...')
        if op.exists(op.join(self.basedir, 'otsu_keras_features.txt')) is False:
            model = InceptionV3(weights='imagenet', include_top=False)
            for im, ell, sub in self.images():
                # im = image.load_img(im, target_size=(224, 224))
                im = io.imread(im)
                binary = im > thresh
                cleaned = im * binary
                # gray_resized = resize(rgb2gray(cleaned), (299, 299))
                resized = resize(cleaned, (299, 299))
                x = image.img_to_array(resized)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = model.predict(x)
                self.keras_features.append(features)

            self.keras_features = np.array(self.keras_features)

            try:
                with open(op.join(self.basedir, 'otsu_keras_features.txt'), 'wb') as file:
                    pickle.dump(self.keras_features, file)
            except:
                print('KerasFeatures improperly dumped')
                sys.exit(1)
        else:
            try:
                print('Keras descriptors alreardy computed, loading..')
                self.keras_features = pickle.load(open(op.join(self.basedir, 'otsu_keras_features.txt'), 'rb'))
            except:
                print("Issues loading alldescriptors!")
                sys.exit(1)

        for img in self.keras_features:
            self.kfeatures.append(img.flatten())
        self.kfeatures = np.array(self.kfeatures)

        try:
            with open(op.join(self.basedir, 'otsu_kfeatures.txt'), 'wb') as file:
                pickle.dump(self.kfeatures, file)
        except:
            print('KerasFeatures improperly dumped')

        return print('KerasFeatures extracted')

    def HOGextractor(self):
        if op.exists(op.join(self.basedir, 'hogdescriptors.txt')) is False:
            print('Computing HOG descriptors...')
            for im, _, c in self.images():
                im = io.imread(im)
                image = rgb2gray(im)
                fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                    cells_per_block=(1, 1), visualise=False)
                self.hogdescriptors.append(fd)
            try:
                with open(op.join(self.basedir, 'hogdescriptors.txt'), 'wb') as file:
                    pickle.dump(self.hogdescriptors, file)
            except:
                print('HOG descriptors not dumped')
        else:
            try:
                print('HOG descriptors alreardy computed, loading..')
                self.hogdescriptors = pickle.load(open(op.join(self.basedir, 'hogdescriptors.txt'), 'rb'))
            except:
                print("Issues loading hogdescriptors!")
                sys.exit(1)
        print('Hog Descriptor computation complete.')

    def Blob_ORB_extraction(self):
        if op.exists(op.join(self.basedir, 'blobORBdescriptors.txt')) is False:
            print('Computing Blob ORB descriptors...')
            descs_list = []
            for im, ell, sub in self.images():
                img = cv2.imread(im)
                blur = cv2.GaussianBlur(img, (5, 5), 0)
                #Color segmentation
                Z = blur.reshape((-1,3))
                # convert to np.float32
                Z = np.float32(Z)
                # define criteria, number of clusters(K) and apply kmeans()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 16
                ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                # Now convert back into uint8, and make original image
                center = np.uint8(center)
                res = center[label.flatten()]
                res2 = res.reshape((blur.shape))

                b, g, r = cv2.split(res2)
                rgb_img = cv2.merge([r, g, b])

                gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

                # noise removal
                im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                mask = np.zeros(thresh.shape, np.uint8)
                mask2 = np.zeros(thresh.shape, np.bool)
                for c in contours:
                    # if the contour is not sufficiently large, ignore it
                    if cv2.contourArea(c) < 7000:
                        continue
                    cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
                mask = cv2.GaussianBlur(mask, (15, 15), 0)
                mask2[mask < 250] = True
                masked_color = cv2.cvtColor(thresh * mask2, cv2.COLOR_GRAY2BGR)

                orb = cv2.ORB_create(nfeatures=3000)
                kp, descs = orb.detectAndCompute(res2 * masked_color, None)
                descs_list.append(np.array(descs).flatten())

            k = 128
            km = KMeans(k)
            concatenated = np.concatenate(descs_list)
            print('Number of descriptors: {}'.format(
                    len(concatenated)))
            concatenated = concatenated[::64]
            print('Clustering with K-means...')
            km.fit(concatenated)
            for d in descs_list:
                c = km.predict(d)
                self.bloblogdescriptors.append(np.bincount(c, minlength=k))
            self.bloblogdescriptors = np.array(self.bloblogdescriptors, dtype=float)
            try:
                with open(op.join(self.basedir, 'blobORBdescriptors.txt'), 'wb') as file:
                    pickle.dump(self.bloblogdescriptors, file)
            except:
                print('Blob Log descriptors not dumped')
        else:
            try:
                print('Blob Log descriptors alreardy computed, loading..')
                self.bloblogdescriptors = pickle.load(open(op.join(self.basedir, 'blobORBdescriptors.txt'), 'rb'))
            except:
                print("Issues loading blob log descriptors!")
                sys.exit(1)

        print('Blob log Descriptor computation complete.')

    def Otsu_threshold(self):
        if op.exists(op.join(self.basedir, 'otsudescriptors.txt')) is False:
            print('Computing Otsu descriptors...')
            for im, _, c in self.images():
                im = io.imread(im)
                binary = im > thresh
                cleaned = im * binary
                self.otsudescriptors.append(rgb2gray(cleaned).flatten())
            try:
                with open(op.join(self.basedir, 'otsudescriptors.txt'), 'wb') as file:
                    pickle.dump(np.array(self.otsudescriptors), file)
            except:
                print('Otsu descriptors not dumped')
        else:
            try:
                print('Otsu descriptors already computed, loading..')
                self.otsudescriptors = pickle.load(open(op.join(self.basedir, 'otsudescriptors.txt'), 'rb'))
            except:
                print("Issues loading otsudescriptors!")
                sys.exit(1)
        print('Otsu Descriptor computation complete.')

    def main(self):
        # self.wholeImage()
        self.SURFextractor()
        # self.keras_features_extraction()
        # self.HOGextractor()
        # self.Blob_ORB_extraction()
        # self.Otsu_threshold()


if __name__ == '__main__':
    os.chdir(op.dirname(op.abspath(__file__)))
    GetFeatures().main()
    # GetFeatures(basedir=op.join(INTERIM, 'train', 'generated'),
    #             subfol=['train']).main()
