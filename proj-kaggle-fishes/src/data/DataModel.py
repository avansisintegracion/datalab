# -*- coding: utf-8 -*-

import json
import sys
import os.path as op
import os
from glob import glob
import hashlib
import numpy as np
import pandas as pd
import cv2
import time
import pickle
import logging
from logging.handlers import RotatingFileHandler
from collections import Counter
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import KMeans

ROOTFOLDER = '../..'


def logger(path, loglevel, message):
    """
    logger method that outputs to both console and file
    logleverl : critical, error, warning, info or debug
    from Sam & Max : http://sametmax.com/ecrire-des-logs-en-python/
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')

    file_handler = RotatingFileHandler(op.join(path, 'activity.log'), 'a', 1000000, 1)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.DEBUG)
    logger.addHandler(steam_handler)

    getattr(logger, loglevel)(str(message))

def open_dump(path, textfile):
    return pickle.load(open(os.path.join(path, textfile), 'rb'))


def removekey(d, key):
    """Function to remove key from dict"""
    r = dict(d)
    del r[key]
    return r


class objdict(dict):
    """Initial class to create dict like objects"""
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class ProjFolder(objdict):
    """
    A class to define project's subfolders for easy access.
    """
    def __init__(self):
        # level data
        self.datafolder = op.join(ROOTFOLDER, 'data')
        for subfol in ['external', 'interim', 'processed', 'raw']:
            setattr(self,
                    'data_' + subfol,
                    op.join(self.datafolder, subfol))

        # level data external
        for subfol in ['annos']:
            setattr(self,
                    'data_external_' + subfol,
                    op.join(self.data_external, subfol))

        # level data interim
        for subfol in ['train', 'test']:
            setattr(self,
                    'data_interim_' + subfol,
                    op.join(self.data_interim, subfol))

        for subfol in ['crop', 'generated']:
            setattr(self,
                    'data_interim_train_' + subfol,
                    op.join(self.data_interim_train, subfol))

        for subfol in ['train', 'val']:
            setattr(self,
                    'data_interim_train_crop_' + subfol,
                    op.join(self.data_interim_train_crop, subfol))

        # level data processed
        # for subfol in []:
        #     setattr(self,
        #             'data_processed_' + subfol,
        #             op.join(self.data_processed, subfol))

        # level data raw
        for subfol in ['train', 'test']:
            setattr(self,
                    'data_raw_' + subfol,
                    op.join(self.data_raw, subfol))


class CustomSplit():
    """
    Custom split of the data to define boat dependant split
    """
    def __init__(self):
        self.f = ProjFolder()
        self.classes = ['ALB',
                        'BET',
                        'DOL',
                        'LAG',
                        'NoF',
                        'OTHER',
                        'SHARK',
                        'YFT']
        self.images = []
        self.X_train = []
        self.y_train = []
        self.X_train_id = []
        self.X_train_norm = []
        self.y_train_norm = []

    def get_im_cv2(self, path):
        img = cv2.imread(path)
        resized = cv2.resize(img, (32, 32))
        return resized

    def load_train(self):
        start_time = time.time()
        print('Read train images')
        for ci, cl in enumerate(self.classes):
            self.images = glob('{}/{}/*.jpg'.format(self.f.data_raw_train, cl))
        for im in sorted(self.images):
            img = self.get_im_cv2(im)
            self.X_train.append(img)
            self.X_train_id.append(op.basename(im))
            self.y_train.append(cl)
        print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
        return

    def read_and_normalize_train_data(self):
        print('Convert to numpy...')
        self.X_train_norm = np.array(self.X_train, dtype=np.uint8)
        self.y_train_norm = np.array(self.y_train, dtype=np.uint8)

        print('Reshape...')
        self.X_train_norm = self.X_train_norm.transpose((0, 3, 1, 2))

        print('Convert to float...')
        self.X_train_norm = self.X_train_norm.astype('float32')
        self.X_train_norm = self.X_train_norm / 255

        print('Train shape:', self.X_train_norm.shape)
        print(self.X_train_norm.shape[0], 'train samples')
        return

    def main(self):
        self.load_train()
        self.read_and_normalize_train_data()

        data_final = self.X_train_norm.reshape(3777, 3072)
        n_components = 50
        pca = RandomizedPCA(n_components=n_components, whiten=True).fit(data_final)
        x_train_pca = pca.transform(data_final)
        n_boats = 14
        kmeans = KMeans(n_clusters=n_boats, random_state=0).fit(x_train_pca)
        predicted_labels = kmeans.predict(x_train_pca)

        df = pd.DataFrame({'img_file': self.X_train_id,
                           'boat_group': predicted_labels,
                           'labels': self.X_train_norm})

        with open(self.f.data_processed + '/df.txt', 'wb') as file:
            pickle.dump(df, file)

        for cat in range(0, 8):
            tmp = df.loc[df['labels'] == cat]
            print('This is cat : %i' % cat)
            counts = Counter(tmp['boat_group'])
            print(counts)

        # Manual selection of categories that allow a 80/20 split
        # that keeps certain boats from being in the subtrain dataset.
        # CHANGE the boatID everytime the script is run to accomodate
        df_20 = df.loc[df['boat_group'].isin([1, 5, 7, 8, 11])]
        df_80 = df.loc[~df['boat_group'].isin([1, 5, 7, 8, 11])]

        with open(self.f.data_processed + '/df_20.txt', 'wb') as file:
            pickle.dump(df_20, file)

        with open(self.f.data_processed + '/df_80.txt', 'wb') as file:
            pickle.dump(df_80, file)


class ImageFile(objdict):
    """
    :param
    :img abs path to img
    :fishtype class of fish
    :datatype training, validation or test data point
    """
    def __init__(self, img, fishtype, datatype):
        self.imgid = str(hashlib.sha1(img.encode()).hexdigest())
        self.imgname = op.basename(img)
        self.imgpath = op.dirname(img)
        self.fishtype = str(fishtype)
        self.datatype = str(datatype)
        if self.datatype == 'training':
            self.validation = None


class ImageList(objdict):
    """
    Create a list of all images with their properties
    Dump to json format
    """
    def __init__(self):
        self.f = ProjFolder()
        self.classes = ['ALB',
                        'BET',
                        'DOL',
                        'LAG',
                        'NoF',
                        'OTHER',
                        'SHARK',
                        'YFT']
        self.images = list()
        self.training_img = dict()
        self.test_img = dict()
        try:
            with open(self.f.data_processed + '/df_80.txt', 'r') as file:
                self.df_80 = pickle.load(file)
        except:
            print('Could not load df_80 !! Breaking..')
            sys.exit(1)

    def training_images(self):
        """Iterate over all (image,label) pairs"""
        for ci, cl in enumerate(self.classes):
            self.images = glob('{}/{}/*.jpg'.format(self.f.data_raw_train, cl))
            for im in sorted(self.images):
                imf = ImageFile(img=im, fishtype=cl, datatype='training')
                self.training_img[imf.imgid] = removekey(imf, 'imgid')
                if any(self.df_80['img_file'] == imf.imgname):
                    cropped = op.join(self.f.data_interim_train,
                                      'crop',
                                      'train',
                                      cl,
                                      imf.imgname)
                    val = False
                else:
                    val = True
                    cropped = op.join(self.f.data_interim_train,
                                      'crop',
                                      'val',
                                      cl,
                                      imf.imgname)
                self.training_img[imf.imgid].update(dict(imgcrop=cropped,
                                                    validation=val))
        with open(op.join(self.f.data_processed, 'training_images.json'), 'wb') as file:
            json.dump(self.training_img, file, sort_keys=False,
                      indent=4, separators=(',', ': '))
        return self.training_img

    def test_images(self):
        """Iterate over all images in test"""
        self.images = glob('{}/*.jpg'.format(self.f.data_raw_test))
        for im in sorted(self.images):
            imf = ImageFile(img=im, fishtype='unknown', datatype='test')
            self.test_img[imf.imgid] = removekey(imf, 'imgid')
            cropped = op.join(self.f.data_interim_test,
                              'crop',
                              imf.imgname)
            self.test_img[imf.imgid].update(dict(imgcrop=cropped))
        with open(op.join(self.f.data_processed, 'test_images.json'), 'wb') as file:
            json.dump(self.test_img, file, sort_keys=False,
                      indent=4, separators=(',', ': '))
        return self.test_img

    def main(self):
        if op.exists(op.join(self.f.data_processed, 'training_images.json')) is False:
            self.training_images()
        if op.exists(op.join(self.f.data_processed, 'test_images.json')) is False:
            self.test_images()


if __name__ == '__main__':
    os.chdir(op.dirname(op.abspath(__file__)))
    test = ImageList()
    test.main()
