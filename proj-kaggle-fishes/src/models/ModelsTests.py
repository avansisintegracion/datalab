from __future__ import print_function
import os
import os.path as op
import re
import time
import itertools
import json
import pickle
from glob import glob
import numpy as np
import pandas as pd
import mahotas as mh
from mahotas.features import surf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.decomposition import PCA

from src.data import DataModel as dm

INTERIM = '../../data/interim'
PROCESSED = '../../data/processed'


class TestClassifications(object):
    '''Classification optimization'''
    def __init__(self):
        self.f = dm.ProjFolder()
        self.ifeatures = {'o': dict(),
                          'f': op.join(self.f.data_interim_train_crop,
                                       'ifeatures.txt')}
        self.sfeatures = {'o': dict(),
                          'f': op.join(self.f.data_interim_train_crop,
                                       'sfeatures.txt')}
        with open(op.join(self.f.data_processed, 'training_images.json'), 'rb') as file:
            self.training_img = json.load(file)
        try:
            print('load trainring')
            self.ifeatures['o'] = pickle.load(open(self.ifeatures['f'], 'rb'))
            self.sfeatures['o'] = pickle.load(open(self.sfeatures['f'], 'rb'))
            self.ifeatures['p'] = pd.DataFrame.from_dict(self.ifeatures['o'],
                                                         orient='index')
            self.sfeatures['p'] = pd.DataFrame.from_dict(self.sfeatures['o'],
                                                         orient='index')
            self.features = pd.concat([self.ifeatures['p'], self.sfeatures['p']], axis=1)
            # self.hogfeatures = open_dump(INTERIM, 'train/crop/hogdescriptors.txt')
            # self.bloblogdescriptors = open_dump(INTERIM, 'train/crop/bloblogdescriptors.txt')
            # self.labels = open_dump(INTERIM, 'train/crop/labels.txt')
            # self.filenames = open_dump(INTERIM, 'train/crop/filenames.txt')
            # self.otsudescriptors = open_dump(INTERIM, 'train/crop/otsudescriptors.txt')
            # self.kfeatures = open_dump(INTERIM, 'train/crop/otsu_kfeatures.txt')
            # self.features = np.hstack([self.kfeatures, self.ifeatures])
            # self.features = np.hstack([self.ifeatures, self.sfeatures])
            # self.features = np.hstack([np.array(self.bloblogdescriptors), self.ifeatures])
            # self.features = np.hstack([np.array(self.hogfeatures), self.ifeatures])
            # self.features = self.otsudescriptors
            # self.df_80 = open_dump(PROCESSED, 'df_80.txt')
            # self.features = self.ifeatures
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

    def split_data(self):
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
        scaler_class.fit(self.X_train, self.y_train)
        y_true, y_pred = self.y_val, scaler_class.predict_proba(self.X_val)
        y_pred = np.clip(y_pred, 0.02, 0.98, out=None)
        return str(classifier), log_loss(y_true, y_pred), scaler_class.predict(self.X_val)

    def testAll(self):
        # ## LogisticRegression
        # C_range = 10.0 ** np.arange(-4, 3)
        # self.CustomGridSearch(preproc=StandardScaler(),
        #                       classifier=LogisticRegression(),
        #                       param_grid={'classifier__C': C_range}
        #                       )
        # classifier = LogisticRegression(C=10)
        # results = self.RunOptClassif(preproc=StandardScaler(),
        #                              classifier=classifier)
        # print("Used :%s" % results[0])
        # print("Logloss score on validation set : %s" % results[1])
        # cnf_matrix = confusion_matrix(self.y_val, results[2])
        # np.set_printoptions(precision=2)
        # # Plot normalized confusion matrix
        # plt.figure()
        # self.plot_confusion_matrix(cnf_matrix,
        #                            classes=self.classes,
        #                            normalize=False,
        #                            title='Confusion matrix')
        # plt.savefig(os.path.join(INTERIM, 'LogisticReg_confusion_matrix.png'),
        #             bbox_inches='tight')
        # #######################################################################
        # ## RandomForest
        # N_range = [10, 30, 50, 70, 90]
        # self.CustomGridSearch(preproc=StandardScaler(),
        #                       classifier=RandomForestClassifier(),
        #                       param_grid={'classifier__n_estimators': N_range}
        #                       )
        # classifier = RandomForestClassifier(n_estimators=50)
        # results = self.RunOptClassif(preproc=StandardScaler(),
        #                              classifier=classifier)
        # print("Used :%s" % results[0])
        # print("Logloss score on validation set : %s" % results[1])
        # cnf_matrix = confusion_matrix(self.y_val, results[2])
        # np.set_printoptions(precision=2)
        # # Plot normalized confusion matrix
        # plt.figure()
        # self.plot_confusion_matrix(cnf_matrix,
        #                            classes=self.classes,
        #                            normalize=False,
        #                            title='Confusion matrix')
        # plt.savefig(os.path.join(INTERIM, 'RandomForest_confusion_matrix.png'),
        #             bbox_inches='tight')
        #######################################################################
        ## Naive bayes
        # alpha = 10.0 ** np.arange(-4, 3)
        # self.CustomGridSearch(preproc=MinMaxScaler(),
        #                       classifier=MultinomialNB(),
        #                       param_grid={'classifier__alpha': alpha}
        #                       )
        # classifier = MultinomialNB(alpha=10)
        # results = self.RunOptClassif(preproc=MinMaxScaler(),
        #                              classifier=classifier)
        # print("Used :%s" % results[0])
        # print("Logloss score on validation set : %s" % results[1])
        # cnf_matrix = confusion_matrix(self.y_val, results[2])
        # np.set_printoptions(precision=2)
        # # Plot normalized confusion matrix
        # plt.figure()
        # self.plot_confusion_matrix(cnf_matrix,
        #                            classes=self.classes,
        #                            normalize=False,
        #                            title='Confusion matrix')
        # plt.savefig(os.path.join(INTERIM, 'NaiveBayes_confusion_matrix.png'),
        #             bbox_inches='tight')
        #######################################################################
        ## xgboost
        # param_test = {'classifier__max_depth': range(3, 10, 2),
        #               'classifier__min_child_weight': range(1, 8, 2),
        #               'classifier__learning_rate': [0.001, 0.1, 0.7, 1],
        #               'classifier__n_estimators': [10, 30, 70, 100, 150],
        #               }
        # param_test = {'classifier__learning_rate': [0.001, 0.1, 0.7, 1],
        #               'classifier__n_estimators': [10, 30, 70, 100, 150],
        #               }
        # self.CustomGridSearch(preproc=PCA(n_components=40, svd_solver='randomized'),
        #                       classifier=xgb.XGBClassifier(objective='multi:softmax'),
        #                       param_grid=param_test
        #                       )
        classifier = xgb.XGBClassifier(learning_rate=0.05,
                                       n_estimators=100,
                                       max_depth=7,
                                       min_child_weight=1,
                                       gamma=0.6,
                                       reg_alpha=0.4,
                                       subsample=0.6,
                                       colsample_bytree=0.8,
                                       objective='multi:softmax',
                                       nthread=6,
                                       scale_pos_weight=1,
                                       seed=70)
        results = self.RunOptClassif(preproc=StandardScaler(),
                                     classifier=classifier)
        # results = self.RunOptClassif(preproc=PCA(n_components=40, svd_solver='randomized'),
        #                              classifier=classifier)
        param = "Parameters used :%s" % results[0]
        resultsscore = "Logloss score on validation set : %s" % results[1]
        cnf_matrix = confusion_matrix(self.y_val, results[2])
        log = param + '\n' + resultsscore + '\nConfusion matrix :\n' + str(cnf_matrix)
        dm.logger(path=self.f.data_interim_train_crop, loglevel='info', message=log)
        # dm.logger(path=self.f.data_interim_train_crop, loglevel='info', message=resultsscore)
        # dm.logger(path=self.f.data_interim_train_crop, loglevel='info', message=cnf_matrix)
        # np.set_printoptions(precision=2)
        # Plot normalized confusion matrix
        # plt.figure()
        # self.plot_confusion_matrix(cnf_matrix,
        #                            classes=self.classes,
        #                            normalize=False,
        #                            title='Confusion matrix')
        # plt.savefig(os.path.join(INTERIM, 'XGBoost_withotsu_confusion_matrix.png'),
        #             bbox_inches='tight')


if __name__ == '__main__':
    os.chdir(op.dirname(op.abspath(__file__)))
    test = TestClassifications()
    test.split_data()
    test.testAll()
