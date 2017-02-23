import time
print(time.ctime()) # Current time
start_time = time.time()

from keras.models import Sequential
from keras.preprocessing import image, sequence
from keras.layers import Flatten, Dense, AveragePooling2D, MaxPooling2D
from keras.layers import ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Activation, Lambda
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.metrics import log_loss, confusion_matrix
from itertools import chain
import itertools
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PATH = "../../data/interim/train/crop/" 
MODELS = "../../models/"

name_classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
img_width = 299
img_height = 299
batch_size = 32
learning_rate = 1e-3
nbr_epoch = 100
nbr_train_samples = 3994 #2872
nbr_val_samples = 905
weights_file_conv = '/home/cristian/.keras/models/vgg16_bn_conv.h5'
weights_file_bn = '/home/cristian/.keras/models/vgg16_bn.h5'

px_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))
def preprocess(x):
    x = x - px_mean
    return x[:, ::-1] # reverse axis bgr->rgb

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
    return


## Transformation for train data
train_datagen = image.ImageDataGenerator(rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

trn_generator = train_datagen.flow_from_directory(
        PATH + 'train',
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        #save_to_dir = './Augm/',
        #save_prefix = 'aug',
        classes = name_classes,
        class_mode = 'categorical')

# Transformationfor val
val_datagen = image.ImageDataGenerator(rescale=1./255)

# Validation set
val_generator = val_datagen.flow_from_directory(
    PATH + 'val',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle = True,
    #save_to_dir = './Augm/',
    #save_prefix = 'aug',
    classes = name_classes,
    class_mode = 'categorical')

## Test
#test_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'test', target_size=(224,224),
#            class_mode=None, shuffle=False, batch_size=1)
#test_np =  np.concatenate([test_batch.next() for i in range(test_batch.nb_sample)])
#test_filenames = test_batch.filenames # Get filenames


# Vgg16 with batch normalization

model = Sequential()
model.add(Lambda(preprocess, input_shape=(3,)+(img_width,img_height)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.load_weights(weights_file_conv)

#model.add(Flatten())
#model.add(Dense(4096, activation='relu'))
#model.add(Dense(4096, activation='relu'))
#model.add(Dense(1000, activation='softmax'))

#model.load_weights(weights_file)
#model.pop(); model.pop(); model.pop()

for layer in model.layers:
    layer.trainable = False


p=0.4
model.add(Flatten())
#model.add(Dense(4096, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(p))
#model.add(Dense(4096, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(p))
model.add(Dense(1000, activation='softmax'))

#model.load_weights(weights_file_bn)

model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=["accuracy"])
#optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

print(model.summary())
earlistop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')
csv_logger = CSVLogger('trainingInceptionCropSmall.log')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, min_lr=1e-5)
SaveModelName = MODELS + "InceptionV3.h5"
best_model = ModelCheckpoint(SaveModelName, monitor='val_acc', verbose = 1, save_best_only = True)
callbacks_list = [earlistop, csv_logger, reduce_lr, best_model]

model.fit_generator(
        trn_generator,
        samples_per_epoch = nbr_train_samples,
        nb_epoch = nbr_epoch,
        validation_data = val_generator,
        nb_val_samples = nbr_val_samples,
        callbacks = callbacks_list)

nbr_augmentation = 5
val_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
for idx in range(nbr_augmentation):
    print('{}th augmentation for testing ...'.format(idx))
    random_seed = np.random.random_integers(0, 100000)

    val_generator = val_datagen.flow_from_directory(
            PATH + 'val',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle = False, # Important !!!
            seed = random_seed,
            classes = None,
            class_mode = None)

    #val_image_list = val_generator.filenames
    #print('image_list: {}'.format(test_image_list[:10]))
    print('Begin to predict for testing data ...')
    if idx == 0:
        preds = model.predict_generator(val_generator, nbr_val_samples)
    else:
        preds += model.predict_generator(val_generator, nbr_val_samples)

preds /= nbr_augmentation

###preds = model.predict_generator(val_generator, nbr_val_samples)

#preds = model.predict(val_np, batch_size=batch_size)

#val_labels_one = [ pred.argmax() for pred in val_labels ]
pred_labels_one = [ pred.argmax() for pred in preds ]

cnf_matrix = confusion_matrix(val_generator.classes, pred_labels_one)

plt.figure()
plot_confusion_matrix(cnf_matrix,
                           classes=name_classes,
                           normalize=False,
                           title='Confusion matrix')
plt.savefig('cmInceptionCropSmall.png',  bbox_inches='tight')
