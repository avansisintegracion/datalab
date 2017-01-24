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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, log_loss, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.metrics import accuracy_score

'''Feature extraction and image preprocessing'''

basedir = os.path.join('..', '..', 'data', 'train')


def features_for(im):
    from features import chist
    im = mh.imread(im)
    img = mh.colors.rgb2grey(im).astype(np.uint8)
    return np.concatenate([mh.features.haralick(img).ravel(),
                           chist(im)])


def images():
    '''Iterate over all (image,label) pairs

    This function will return
    '''
    for ci, cl in enumerate(classes):
        images = glob('{}/{}/*.jpg'.format(basedir, cl))
        for im in sorted(images):
            yield im, ci


classes = ['ALB',
           'BET',
           'DOL',
           'LAG',
           'SHARK',
           'YFT',
           'OTHER',
           'NoF'
           ]

print('Computing whole-image texture features...')
ifeatures = []
labels = []
for im, ell in images():
    ifeatures.append(features_for(im))
    labels.append(ell)

ifeatures = np.array(ifeatures)
labels = np.array(labels)

with open(os.path.join(basedir, 'ifeatures.txt'), 'wb') as file:
    pickle.dump(ifeatures, file)

with open(os.path.join(basedir, 'labels.txt'), 'wb') as file:
    pickle.dump(labels, file)


print('Computing SURF descriptors...')
alldescriptors = []
for im, _ in images():
    im = mh.imread(im, as_grey=True)
    im = im.astype(np.uint8)

    # To use dense sampling, you can try the following line:
    # alldescriptors.append(surf.dense(im, spacing=16))
    alldescriptors.append(surf.surf(im, descriptor_only=True))

with open(os.path.join(basedir, 'alldescriptors.txt'), 'wb') as file:
    pickle.dump(alldescriptors, file)

print('Descriptor computation complete.')
k = 256
km = KMeans(k)


concatenated = np.concatenate(alldescriptors)
print('Number of descriptors: {}'.format(
        len(concatenated)))
concatenated = concatenated[::64]
print('Clustering with K-means...')
km.fit(concatenated)
sfeatures = []
for d in alldescriptors:
    c = km.predict(d)
    sfeatures.append(np.bincount(c, minlength=k))
sfeatures = np.array(sfeatures, dtype=float)

with open(os.path.join(basedir, 'sfeatures.txt'), 'wb') as file:
    pickle.dump(sfeatures, file)

'''Classification optimization'''

# basedir = '/Users/lab/Mikael/data'

ifeatures = pickle.load(open(os.path.join(basedir, 'ifeatures.txt'), 'rb'))
sfeatures = pickle.load(open(os.path.join(basedir, 'sfeatures.txt'), 'rb'))
labels = pickle.load(open(os.path.join(basedir, 'labels.txt'), 'rb'))

allfeatures = np.hstack([sfeatures, ifeatures])

# Optimization of hyperparam for LogisticRegression


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def plot_confusion_matrix(cm, classes,
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

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


X_train, X_test, y_train, y_test = train_test_split(allfeatures,
                                                    labels,
                                                    test_size=0.2,
                                                    stratify=labels)

C_range = 10.0 ** np.arange(-4, 3)
clf = Pipeline([('preproc', StandardScaler()),
                ('classifier', LogisticRegression())])
grid = GridSearchCV(clf,
                    param_grid={'classifier__C': C_range},
                    cv=5,
                    scoring='neg_log_loss')

print('Performing classification with all features combined...')
start = time.time()
grid.fit(X_train, y_train)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time.time() - start, len(grid.cv_results_['params'])))
report(grid.cv_results_)

# Model with rank: 1
# Mean validation score: -0.738 (std: 0.052)
# Parameters: {'classifier__C': 1.0}
#
# Model with rank: 2
# Mean validation score: -0.744 (std: 0.033)
# Parameters: {'classifier__C': 0.10000000000000001}
#
# Model with rank: 3
# Mean validation score: -1.032 (std: 0.110)
# Parameters: {'classifier__C': 10.0}

scaler_logreg = Pipeline([('preproc', StandardScaler()),
                          ('classifier', LogisticRegression(C=1))])
scaler_logreg.fit(X_train, y_train)

y_true, y_pred = y_test, scaler_logreg.predict_proba(X_test)
log_loss(y_true, y_pred)
# 0.66167851601953565

y_pred = scaler_logreg.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False,
                      title='Confusion matrix')
plt.show()


# Test with random forest
N_range = [10, 30, 50, 70, 90]
clf = Pipeline([('preproc', StandardScaler()),
                ('classifier', RandomForestClassifier())])
grid = GridSearchCV(clf,
                    param_grid={'classifier__n_estimators': N_range},
                    cv=5,
                    scoring='neg_log_loss')

print('Performing classification with all features combined...')
start = time.time()
grid.fit(X_train, y_train)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time.time() - start, len(grid.cv_results_['params'])))
report(grid.cv_results_)

# Model with rank: 1
# Mean validation score: -0.502 (std: 0.010)
# Parameters: {'classifier__n_estimators': 70}
#
# Model with rank: 2
# Mean validation score: -0.514 (std: 0.029)
# Parameters: {'classifier__n_estimators': 90}
#
# Model with rank: 3
# Mean validation score: -0.559 (std: 0.054)
# Parameters: {'classifier__n_estimators': 50}

scaler_rdnfor = Pipeline([('preproc', StandardScaler()),
                          ('classifier', RandomForestClassifier(n_estimators=70))])
scaler_rdnfor.fit(X_train, y_train)
y_true, y_pred = y_test, scaler_rdnfor.predict_proba(X_test)
log_loss(y_true, y_pred)
# 0.52833125901143274

y_pred = scaler_rdnfor.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False,
                      title='Confusion matrix')
plt.show()


# Test with naive bayes
scaler_multinomialNB = Pipeline([('preproc', MinMaxScaler()),
                                 ('classifier', MultinomialNB(alpha=0.001))])
scaler_multinomialNB.fit(X_train, y_train)
y_true, y_pred = y_test, scaler_multinomialNB.predict_proba(X_test)
log_loss(y_true, y_pred)
# 2.2503639100682293 --> not a good idea

# sklearn.ensemble.GradientBoostingClassifier
scaler_GDBoost = Pipeline([('preproc', StandardScaler()),
                ('classifier', GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0))])
scaler_GDBoost.fit(X_train, y_train)
y_true, y_pred = y_test, scaler_GDBoost.predict_proba(X_test)
log_loss(y_true, y_pred)
# 2.4354344058403989

# testing with xgboost
param_test1 = {'max_depth': range(3, 10, 2),
               'min_child_weight': range(1, 6, 2)
               }

gsearch1 = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate=0.1,
                                                    n_estimators=100,
                                                    max_depth=5,
                                                    min_child_weight=1,
                                                    gamma=0,
                                                    subsample=0.8,
                                                    colsample_bytree=0.8,
                                                    objective='multi:softmax',
                                                    nthread=4,
                                                    scale_pos_weight=1,
                                                    seed=27),
                        param_grid=param_test1,
                        scoring='neg_log_loss',
                        n_jobs=4,
                        iid=False,
                        cv=5)

gsearch1.fit(X_train, y_train)

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

model = xgb.XGBClassifier(max_depth=1, objective='multi:softmax',
                          reg_lambda=0.1, subsample=0.8)

model.fit(X_train, y_train, eval_metric='mlogloss')
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f" % (accuracy))
# Accuracy: 0.64

'''Analysis of test data'''

testdir = os.path.join('..', '..', 'data', 'test_stg1')


def images_test():
    '''Iterate over all image

    This function will return
    '''
    for im in glob('{}/*.jpg'.format(testdir)):
        name = os.path.basename(im)
        yield im, name


tfeatures = []
tnames = []
for im, fname in images_test():
    tfeatures.append(features_for(im))
    tnames.append(fname)

tfeatures = np.array(tfeatures)
tnames = np.array(tnames)

with open(os.path.join(basedir, 'tfeatures.txt'), 'wb') as file:
    pickle.dump(tfeatures, file)

# print('Computing SURF descriptors...')
# talldescriptors = []
# for im,_ in images_test():
#     im = mh.imread(im, as_grey=True)
#     im = im.astype(np.uint8)
#
#     # To use dense sampling, you can try the following line:
#     # alldescriptors.append(surf.dense(im, spacing=16))
#     talldescriptors.append(surf.surf(im, descriptor_only=True))
#
# with open(os.path.join(basedir, 'talldescriptors.txt'), 'wb') as file:
#     pickle.dump(talldescriptors, file)

talldescriptors = pickle.load(open(os.path.join(basedir, 'talldescriptors.txt'),
                                   'rb'))

print('Descriptor computation complete.')
# k = 256
# km = KMeans(k)
#
#
# tconcatenated = np.concatenate(talldescriptors)
# print('Number of descriptors: {}'.format(
#         len(tconcatenated)))
# tconcatenated = tconcatenated[::64]
# print('Clustering with K-means...')
# km.fit(tconcatenated)
# tsfeatures = []
# for d in talldescriptors:
#     c = km.predict(d)
#     tsfeatures.append(np.bincount(c, minlength=k))
# tsfeatures = np.array(tsfeatures, dtype=float)
#
# with open(os.path.join(basedir, 'tsfeatures.txt'), 'wb') as file:
#     pickle.dump(tsfeatures, file)
#
tsfeatures = pickle.load(open(os.path.join(basedir, 'tsfeatures.txt'), 'rb'))

tallfeatures = np.hstack([tsfeatures, tfeatures])

y_pred_test = scaler_rdnfor.predict_proba(tallfeatures)

pd.DataFrame(y_pred_test,
             index=tnames,
             columns=['ALB',
                      'BET',
                      'DOL',
                      'LAG',
                      'SHARK',
                      'YFT',
                      'OTHER',
                      'NoF']).to_csv(os.path.join(testdir, 'predictions.csv'))
