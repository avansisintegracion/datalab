from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler, SMOTE


# Apply the random over-sampling
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(test.X_train, test.y_train)

RANDOM_STATE = 42
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
pipeline = make_pipeline(NearMiss(version=3, random_state=RANDOM_STATE),
                         StandardScaler(),
                         classifier)
pipeline.fit(test.X_train, test.y_train)
cnf_matrix = confusion_matrix(test.y_val, pipeline.predict(test.X_val))
print(cnf_matrix)
print(classification_report_imbalanced(test.y_val, pipeline.predict(test.X_val), target_names=test.classes))
y_true, y_pred = test.y_val, pipeline.predict_proba(test.X_val)
print(log_loss(y_true, y_pred))

np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
test.plot_confusion_matrix(cnf_matrix,
                           classes=test.classes,
                           normalize=False,
                           title='Confusion matrix with NearMiss v2')
plt.savefig(os.path.join(INTERIM, 'XGBoost_NearMiss_confusion_matrix.png'),
            bbox_inches='tight')


pipeline = make_pipeline(StandardScaler(),
                         classifier)
pipeline.fit(test.X_train, test.y_train)
cnf_matrix = confusion_matrix(test.y_val, pipeline.predict(test.X_val))
print(cnf_matrix)
print(classification_report_imbalanced(test.y_val, pipeline.predict(test.X_val), target_names=test.classes))
y_true, y_pred = test.y_val, pipeline.predict_proba(test.X_val)
print(log_loss(y_true, y_pred))

np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
test.plot_confusion_matrix(cnf_matrix,
                           classes=test.classes,
                           normalize=False,
                           title='Confusion matrix')
plt.savefig(os.path.join(INTERIM, 'XGBoost_confusion_matrix.png'),
            bbox_inches='tight')





from imblearn.over_sampling import RandomOverSampler


# Apply the random over-sampling
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(test.X_train, test.y_train)

pipeline = make_pipeline(StandardScaler(),
                         classifier)
pipeline.fit(X_resampled, y_resampled)
cnf_matrix = confusion_matrix(test.y_val, pipeline.predict(test.X_val))
print(cnf_matrix)
print(classification_report_imbalanced(test.y_val, pipeline.predict(test.X_val), target_names=test.classes))
y_true, y_pred = test.y_val, pipeline.predict_proba(test.X_val)
log_loss(y_true, y_pred)

np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
test.plot_confusion_matrix(cnf_matrix,
                           classes=test.classes,
                           normalize=False,
                           title='Confusion matrix with RandomSampleOver')
plt.savefig(os.path.join(INTERIM, 'XGBoost_RSO_confusion_matrix.png'),
            bbox_inches='tight')


from imblearn.under_sampling import (EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours)
# Create the samplers
enn = EditedNearestNeighbours()
renn = RepeatedEditedNearestNeighbours()
pipeline = make_pipeline(enn, renn, StandardScaler(), classifier)
pipeline.fit(test.X_train, test.y_train)
cnf_matrix = confusion_matrix(test.y_val, pipeline.predict(test.X_val))
print(cnf_matrix)
print(classification_report_imbalanced(test.y_val, pipeline.predict(test.X_val), target_names=test.classes))
y_true, y_pred = test.y_val, pipeline.predict_proba(test.X_val)
print(log_loss(y_true, y_pred))


from sklearn.svm import SVC

RANDOM_STATE = 42
pipeline = make_pipeline(NearMiss(version=2, random_state=RANDOM_STATE),
                         StandardScaler(),
                         SVC(probability=True))
pipeline.fit(test.X_train, test.y_train)
cnf_matrix = confusion_matrix(test.y_val, pipeline.predict(test.X_val))
print(cnf_matrix)
print(classification_report_imbalanced(test.y_val, pipeline.predict(test.X_val), target_names=test.classes))
y_true, y_pred = test.y_val, pipeline.predict_proba(test.X_val)
print(log_loss(y_true, y_pred))










from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
subclasses = ['BET',
              'DOL',
              'LAG',
              'SHARK',
              'YFT',
              ]
INTERIM = '../../data/interim/train/crop'


def images():
    '''Iterate over all (image,label) pairs'''
    for ci, cl in enumerate(subclasses):
        images = glob('{}/train/{}/*.jpg'.format(INTERIM, cl))
        for im in sorted(images):
            yield im, cl, str(im.split('/')[-1])


rdnGeneratedImg = '../../data/interim/train/generated/train'

for im, cl, filename in images():
    img = load_img(im)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    genimgsubfol = os.path.join(rdnGeneratedImg, cl)
    if not os.path.exists(genimgsubfol):
        os.makedirs(genimgsubfol)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=genimgsubfol,
                              save_prefix=filename.rstrip('.jpg'),
                              save_format='jpeg'):
        i += 1
        if i > 8:
            break  # otherwise the generator would loop indefinitely


def images():
    '''Iterate over all (image,label) pairs'''
    for ci, cl in enumerate(subclasses):
        images = glob('{}/train/{}/*.jpg'.format(INTERIM, cl))
        for im in sorted(images):
            yield im, cl, str(im.split('/')[-1])

def images(self):
    '''Iterate over all (image,label) pairs'''
    for ci, cl in enumerate(self.classes):
        try:
            for fol in self.subfol:
                images = glob('{}/{}/{}/*.jpg'.format(self.basedir, fol, cl))
                for im in sorted(images):
                    yield im, ci, str(im.split('/')[-3])
        except:
            continue
