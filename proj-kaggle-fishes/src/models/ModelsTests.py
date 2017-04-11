"""
"""

# -*- coding: utf-8 -*-

import os
import os.path as op
import time
import itertools
import json
import numpy as np
import pandas as pd
import pickle as p

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from skimage import io
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, confusion_matrix

import mahotas as mh
from mahotas.features import surf
from sklearn.cluster import KMeans
import cv2


from src.data import DataModel as dm

INTERIM = '../../data/interim'
PROCESSED = '../../data/processed'


def open_dump(path, textfile):
    return p.load(open(os.path.join(path, textfile), 'rb'))


class TestClassifications(object):
    '''Classification optimization'''
    def __init__(self, ifeatures, sfeatures, projectfolder, imagetype):
        self.f = projectfolder
        self.ifeatures = ifeatures
        self.sfeatures = sfeatures
        self.imagetype = imagetype
        with open(op.join(self.f.data_processed, 'training_images.json'), 'rb') as file:
            self.training_img = json.load(file)
        try:
            self.ifeatures['o'] = p.load(open(self.ifeatures['f'], 'rb'))
            self.sfeatures['o'] = p.load(open(self.sfeatures['f'], 'rb'))
            self.ifeatures['p'] = pd.DataFrame.from_dict(self.ifeatures['o'],
                                                         orient='index')
            self.sfeatures['p'] = pd.DataFrame.from_dict(self.sfeatures['o'],
                                                         orient='index')
            self.features = pd.concat([self.ifeatures['p'], self.sfeatures['p']], axis=1)
            print(self.sfeatures['f'])
        except:
            print('An error occured during loading of data')
        self.X_train = []
        self.y_train = []
        self.X_val = []
        self.y_val = []
        self.classes = ['ALB',
                        'BET',
                        'DOL',
                        'LAG',
                        'NoF',
                        'OTHER',
                        'SHARK',
                        'YFT']

    def report(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
        return

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return

    def split_data_random(self):
        """
        This method aims at randomly splitting between training/validation
        using the classical train_test_split function.
        """
        X_data = list()
        y_data = list()
        for k, img in self.training_img.iteritems():
                X_data.append(self.features.loc[k, ].values)
                y_data.append(img['fishtype'])
        X_data = np.array(X_data)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_data, y_data, test_size=0.4, random_state=0)
        return

    def split_data(self):
        """
        This method uses custom designed validation tag in order to split
        images taking into account the boat status, ie. validation set contains
        pictures from boats that were never seen in the training set.
        """
        for k, img in self.training_img.iteritems():
            if img['validation'] is False:
                self.X_train.append(self.features.loc[k, ].values)
                self.y_train.append(img['fishtype'])
            else:
                self.X_val.append(self.features.loc[k, ].values)
                self.y_val.append(img['fishtype'])

        self.X_train = np.array(self.X_train)
        self.X_val = np.array(self.X_val)
        return

    def split_data_img(self):
        """
        This method uses custom designed validation tag in order to split
        images taking into account the boat status, ie. validation set contains
        pictures from boats that were never seen in the training set.
        To be used for convnet where we need pictures directly into the net.
        """
        for k, img in self.training_img.iteritems():
            if self.imagetype == 'raw':
                im = resize(io.imread(op.join(img['imgpath'], img['imgname']), as_grey=True), (250, 250)).ravel()
            else:
                im = resize(io.imread(img[self.imagetype], as_grey=True), (250, 250)).ravel()
            if img['validation'] is False:
                self.X_train.append(im)
                self.y_train.append(img['fishtype'])
            else:
                self.X_val.append(self.features.loc[k, ].values)
                self.y_val.append(img['fishtype'])

        self.X_train = np.array(self.X_train)
        self.X_val = np.array(self.X_val)
        return

    def CustomGridSearch(self, preproc, classifier, param_grid):
        clf = Pipeline([('preproc', preproc),
                        ('classifier', classifier)])
        grid = GridSearchCV(clf,
                            param_grid=param_grid,
                            cv=5,
                            scoring='neg_log_loss')
        print('Performing classification with %s...' % str(classifier))
        start = time.time()
        grid.fit(self.X_train, self.y_train)
        print("GridSearchCV took %.2f sec for %d candidate parameter settings."
              % (time.time() - start, len(grid.cv_results_['params'])))
        return self.report(grid.cv_results_)

    def RunOptClassif(self, preproc, classifier):
        scaler_class = Pipeline([('preproc', preproc),
                                 ('classifier', classifier)])
        scaler_class.fit(self.X_train,
                         self.y_train,
                         classifier__eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
                         classifier__early_stopping_rounds=50)
        y_true, y_pred = self.y_val, scaler_class.predict_proba(self.X_val)
        # y_pred = np.clip(y_pred, 0.02, 0.98, out=None)
        return str(classifier), log_loss(y_true, y_pred), scaler_class.predict(self.X_val), scaler_class

    def train(self):
        ## xgboost
        # param_test = {'classifier__max_depth': range(3, 10, 2),
        #               'classifier__min_child_weight': range(1, 8, 2),
        #               'classifier__learning_rate': [0.001, 0.1, 0.7, 1],
        #               'classifier__n_estimators': [10, 30, 70, 100, 150],
        #               }
        # param_test = {'preproc__learning_rate': [0.001, 0.1, 0.7, 1],
        #               'classifier__n_estimators': [10, 30, 70, 100, 150],
        #               }
        # param_test = {'classifier__learning_rate': [0.001, 0.1, 0.7, 1],
        #               'classifier__n_estimators': [10, 30, 70, 100, 150],
        #               }
        # param_test = {'classifier__n_estimators': [100, 150, 200, 250, 300],
        #               }
        # # PCA(n_components=40, svd_solver='randomized')
        # results = self.CustomGridSearch(preproc=StandardScaler(),
        #                       classifier=xgb.XGBClassifier(objective='multi:softmax'),
        #                       param_grid=param_test
        #                       )
        # log = results
        # dm.logger(path=self.f.data_interim_train_crop, loglevel='info', message=log

        classifier = xgb.XGBClassifier(learning_rate=0.1,
                                       n_estimators=400,
                                       max_depth=9,
                                       min_child_weight=1,
                                       objective='multi:softmax',
                                       reg_alpha=0.5,
                                       gamma=5)
        results = self.RunOptClassif(preproc=StandardScaler(),
                                     classifier=classifier)
        param = "Parameters used :%s" % results[0]
        resultsscore = "Logloss score on validation set : %s" % results[1]
        cnf_matrix = confusion_matrix(self.y_val, results[2])
        log = param + '\n' + resultsscore + '\nConfusion matrix :\n' + str(cnf_matrix)
        dm.logger(path=self.f.data_interim_train_raw, loglevel='info', message=log)
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix,
                                   classes=self.classes,
                                   normalize=False,
                                   title='Confusion matrix')
        plt.savefig(os.path.join(INTERIM, 'XGBoost_confusion_matrix_rotatecrop_images.png'),
                    bbox_inches='tight')

        model = results[3]
        with open(op.join(self.f.data_processed, 'xgboost.model'), 'wb') as file:
            p.dump(model, file)
        return model


class predictTestImages():
    def __init__(self, projectfolder, imagetype, model):
        self.f = projectfolder
        self.imagetype = imagetype
        with open(op.join(self.f.data_processed, 'test2_images.json'), 'rb') as file:
            self.test2_img = json.load(file)
        self.classes = ['ALB',
                        'BET',
                        'DOL',
                        'LAG',
                        'NoF',
                        'OTHER',
                        'SHARK',
                        'YFT']
        self.model = model
        with open(op.join(self.f.data_interim_train_raw, 'kmodel.dump'), 'rb') as file:
            self.kmodel = p.load(file)

    def predict(self):
        def chist(im):
            im = im // 64
            r, g, b = im.transpose((2, 0, 1))
            pixels = 1 * r + 4 * g + 16 * b
            hist = np.bincount(pixels.ravel(), minlength=64)
            hist = hist.astype(float)
            return np.log1p(hist)

        def features_for(im):
            """Extract color histogram from image"""
            im = mh.imread(im)
            img = mh.colors.rgb2grey(im).astype(np.uint8)
            return np.concatenate([mh.features.haralick(img).ravel(),
                                   chist(im)])

        def wholeImage(im):
            return np.array(features_for(im))

        def SURFextractor(im):
            img = cv2.imread(im)
            Z = img.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 16
            ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((img.shape))
            gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
            if gray.size < 10000:
                gray = cv2.resize(gray, (200, 200))
            alldescriptor = surf.surf(gray, descriptor_only=True)
            c = self.kmodel.predict(alldescriptor)
            sfeatures = np.bincount(c, minlength=128)
            return sfeatures

        features = list()
        for k, img in self.test2_img.iteritems():
            im = op.join(img['imgpath'], img['imgname'])
            feat = np.hstack((wholeImage(im), SURFextractor(im)))
            print feat.shape
            features.append(feat)

        features = np.vstack(features)
        print features
        with open(op.join(self.f.data_interim_train_raw, 'features_test2.dump'), 'wb') as file:
            p.dump(features, file)

        self.model.predict_proba(features)
        return


if __name__ == '__main__':
    os.chdir(op.dirname(op.abspath(__file__)))
    projectfolder = dm.ProjFolder()
    test = TestClassifications(ifeatures={'o': dict(),
                                          'f': op.join(projectfolder.data_interim_train_raw, 'fifeatures.txt')},
                               sfeatures={'o': dict(),
                                          'f': op.join(projectfolder.data_interim_train_raw, 'fsfeatures.txt')},
                               projectfolder=projectfolder,
                               imagetype='raw')
    test.split_data_random()
    # test.split_data_img()
    # test.split_data()
    model = test.train()
    predict_test2 = predictTestImages(projectfolder=projectfolder,
                                      imagetype='raw',
                                      model=model)
    predict_test2.predict()
