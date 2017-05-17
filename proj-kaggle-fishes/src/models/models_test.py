from glob import glob
import re
from __future__ import print_function

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mahotas as mh
from mahotas.features import surf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import log_loss, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM
from scipy.ndimage import convolve


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

        # rbm = BernoulliRBM(random_state=0, verbose=True)
        # rbm.learning_rate = 0.06
        # rbm.n_iter = 20
        # # More components tend to give better prediction performance, but larger
        # # fitting time
        # rbm.n_components = 200
        # gamma=0.6,
        # reg_alpha=0.4,
        # subsample=0.6,
        # colsample_bytree=0.8,
        # scale_pos_weight=1,
        # seed=70

        # plt.figure(figsize=(4.2, 4))
        # for i, comp in enumerate(rbm.components_):
        #     plt.subplot(10, 20, i + 1)
        #     plt.imshow(comp.reshape((224, 224)), cmap=plt.cm.gray_r,
        #                interpolation='nearest')
        #     plt.xticks(())
        #     plt.yticks(())
        # plt.suptitle('200 components extracted by RBM', fontsize=16)
        # plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        # plt.savefig(os.path.join(self.f.data_interim_train_crop, 'XGBoost_RBM_features.png'),
        #             bbox_inches='tight')
        # dm.logger(path=self.f.data_interim_train_crop, loglevel='info', message=resultsscore)
        # dm.logger(path=self.f.data_interim_train_crop, loglevel='info', message=cnf_matrix)
        # np.set_printoptions(precision=2)
        # Plot normalized confusion matrix

        # results = self.RunOptClassif(preproc=rbm,
        #                      classifier=classifier)

        # results = self.RunOptClassif(preproc=PCA(n_components=40, svd_solver='randomized'),
        #                              classifier=classifier)
