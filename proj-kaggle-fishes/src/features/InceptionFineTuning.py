import time
print(time.ctime()) # Current time
start_time = time.time()

# Functions and basic import libraries in the utils.py file
#from utils import *
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
#from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, AveragePooling2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image, sequence
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.np_utils import to_categorical
from sklearn.metrics import log_loss, confusion_matrix
from itertools import chain
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import os
from IPython.core.debugger import Tracer
K.set_image_dim_ordering('tf')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
PATH = "../../data/interim/train/crop/" 
MODELS = "../../models/"
PROCESSED = "../../data/processed/"
ANNOS = "../../data/external/annos/"


### Load data
## Train

## Train read from batch 
#def LoadData():
#    trn_batch = trn_generator.flow_from_directory(PATH + 'train', target_size=(224,224),
#                class_mode=None, shuffle=False, batch_size=1)
#    trn_np =  np.concatenate([trn_batch.next() for i in range(trn_batch.nb_sample)])
#    filenames = trn_batch.filenames # Get filenames
#    #trn_labels = to_categorical(trn_batch.classes) # Classes to one-hot encoding
#
#    # Validation set
#    val_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'val', target_size=(224,224),
#                class_mode=None, shuffle=False, batch_size=1)
#    val_np =  np.concatenate([val_batch.next() for i in range(val_batch.nb_sample)])
#    val_filenames = val_batch.filenames # Get filenames
#    #val_labels = to_categorical(val_batch.classes) # Classes to one-hot encoding
#
#    ## Test
#    #test_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'test', target_size=(224,224),
#    #            class_mode=None, shuffle=False, batch_size=1)
#    #test_np =  np.concatenate([test_batch.next() for i in range(test_batch.nb_sample)])
#    #test_filenames = test_batch.filenames # Get filenames
#
#    base_model = InceptionV3(weights='imagenet', include_top=False)
#    bottleneck_features_train = base_model.predict(trn_np)
#    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
#
#    bottleneck_features_validation = base_model.predict(val_np)
#    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
#    print("--- Save Features  %s seconds ---" % (time.time() - start_time))
#
### Bounding boxes & multi output
#def LoadJsonLabels():
#    import ujson as json
#    anno_classes = ['alb', 'bet', 'dol', 'lag', 'shark', 'yft']
#
#    bb_json = {}
#    for c in anno_classes:
#        j = json.load(open('{}{}_labels.json'.format(ANNOS, c), 'r'))
#        for l in j:
#            if 'annotations' in l.keys() and len(l['annotations'])>0:
#                bb_json[l['filename'].split('/')[-1]] = sorted(
#                    l['annotations'], key=lambda x: x['height']*x['width'])[-1]
#
#    # Get python raw filenames (without foldername)
#    raw_filenames = [f.split('/')[-1] for f in filenames]
#    raw_val_filenames = [f.split('/')[-1] for f in val_filenames]
#    raw_test_filenames = [f.split('/')[-1] for f in test_filenames]
#
#    # Image that have no annotation, empty bounding box
#    empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}
#    for f in raw_filenames:
#        if not f in bb_json.keys(): bb_json[f] = empty_bbox
#    for f in raw_test_filenames:
#        if not f in bb_json.keys(): bb_json[f] = empty_bbox
#     
#    # The sizes of images can be related to ship sizes. Get sizes for raw image
#    sizes = [PIL.Image.open(PATH+'train/'+f).size for f in filenames]
#    val_sizes = [PIL.Image.open(PATH+'val/'+f).size for f in val_filenames]
#    raw_test_sizes = [PIL.Image.open(PATH+'test/'+f).size for f in test_filenames]
#
#    # Convert dictionary into array
#    bb_params = ['height', 'width', 'x', 'y']
#    def convert_bb(bb, size):
#        bb = [bb[p] for p in bb_params]
#        conv_x = (224. / size[0])
#        conv_y = (224. / size[1])
#        bb[0] = max((bb[0]+bb[3])*conv_y, 0)
#        bb[1] = max((bb[1]+bb[2])*conv_x, 0)
#        bb[2] = max(bb[2]*conv_x, 0)
#        bb[3] = max(bb[3]*conv_y, 0)
#        return bb
#
#    # Tranform box in terms of defined compressed image size (224)
#    trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, sizes)],).astype(np.float32)
#
#    print("--- Bounding boxes reading  %s seconds ---" % (time.time() - start_time))


#def TrainTopModel():
#    train_data = np.load(open('bottleneck_features_train.npy'))
#    trn_batch = trn_generator.flow_from_directory(PATH + 'train', target_size=(224,224),
#                class_mode=None, shuffle=False, batch_size=1)
#    trn_labels = to_categorical(trn_batch.classes) # Classes to one-hot encoding
#    filenames = trn_batch.filenames # Get filenames
#
#    validation_data = np.load(open('bottleneck_features_validation.npy'))
#    val_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'valid', target_size=(224,224),
#                class_mode=None, shuffle=False, batch_size=1)
#    val_labels = to_categorical(val_batch.classes) # Classes to one-hot encoding
#    filenames = val_batch.filenames # Get filenames
#
#    model = Sequential()
#    model.add(Flatten(input_shape=train_data.shape[1:]))
#    model.add(Dense(256, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(8, activation='sigmoid'))
#
#    earlistop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
#    csv_logger = CSVLogger('training.log')
#    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, min_lr=1e-5)
#    callbacks_list = [earlistop,csv_logger, reduce_lr]
#    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#
#    model.fit(train_data, trn_labels, nb_epoch=50, batch_size=64,
#              validation_data=(validation_data, val_labels), callbacks=callbacks_list)
#    model.save_weights(MODELS + "top_model_fc.h5")
#
#    print("--- Train top model %s seconds ---" % (time.time() - start_time))

def ProbabilityDistribution(classes):
    blasses_names = list(classes)
    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharex='col', sharey='row')
    ax1.hist(classes["ALB"],bins=20)
    ax1.set_title("ALB")
    ax2.hist(classes["BET"],bins=20)
    ax2.set_title("BET")
    ax3.hist(classes["DOL"],bins=20)
    ax3.set_title("DOL")
    ax3.set_xticks(np.arange(0,1.2,0.2))
    ax3.set_xlim(0, 1)
    ax4.hist(classes["LAG"],bins=20)
    ax4.set_title("LAG")
    ax5.hist(classes["NoF"],bins=20)
    ax5.set_title("NoF")
    ax6.hist(classes["OTHER"],bins=20)
    ax6.set_title("OTHER")
    ax7.hist(classes["SHARK"],bins=20)
    ax7.set_title("SHARK")
    ax7.set_xticks(np.arange(0,1.2,0.2))
    ax8.hist(classes["YFT"],bins=20)
    ax8.set_title("YFT")
    name=PATH + 'pdInceptionCropSmall.png'
    plt.savefig(name)
    plt.cla()


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



def FineTuning():
    name_classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    img_width = 299
    img_height = 299
    batch_size = 32
    learning_rate = 1e-3
    nbr_epoch = 100
    nbr_train_samples = 3994 #2872
    nbr_val_samples = 905

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
    #trn_batch = trn_generator.flow_from_directory(PATH + 'train', target_size=(224,224),
    #            class_mode=None, shuffle=False, batch_size=1)
    #trn_np =  np.concatenate([trn_batch.next() for i in range(trn_batch.nb_sample)])
    #trn_labels = to_categorical(trn_batch.classes) # Classes to one-hot encoding
    #filenames = trn_batch.filenames # Get filenames

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
    #val_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'val', target_size=(224,224),
    #            class_mode=None, shuffle=False, batch_size=1)
    #val_np =  np.concatenate([val_batch.next() for i in range(val_batch.nb_sample)])
    #val_labels = to_categorical(val_batch.classes) # Classes to one-hot encoding
    #val_filenames = val_batch.filenames # Get filenames


    ## Test
    #test_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'test', target_size=(224,224),
    #            class_mode=None, shuffle=False, batch_size=1)
    #test_np =  np.concatenate([test_batch.next() for i in range(test_batch.nb_sample)])
    #test_filenames = test_batch.filenames # Get filenames

    # build a classifier model to put on top of the convolutional model
    # add a global spatial average pooling layer

    #input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_dim_ordering() == 'tf'
    print('Loading InceptionV3 Weights ...')
    base_model = InceptionV3(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape=(299, 299, 3))
    # Note that the preprocessing of InceptionV3 is:
    # (x / 255 - 0.5) x 2

    print('Adding Average Pooling Layer and Softmax Output Layer ...')
    output = base_model.get_layer(index = -1).output  # Shape: (8, 8, 2048)
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    #output = Dense(512, activation='relu')(output)
    output = Dense(8, activation='softmax', name='predictions')(output)

    model = Model(base_model.input, output)

    ## first: train only the top layers (which were randomly initialized)
    ## i.e. freeze all convolutional InceptionV3 layers
    #for layer in model.layers[:211]:
    #   layer.trainable = False
    #for layer in model.layers[211:]:
    #   layer.trainable = True

    optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
    model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    earlistop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')
    csv_logger = CSVLogger(PATH + 'trainingInceptionCropSmall.log')
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

    #model.fit(trn_np, trn_labels, nb_epoch=100, batch_size=batch_size,
    #              validation_data=(val_np, val_labels), callbacks=callbacks_list)

    ## we chose to train the top 2 inception blocks, i.e. we will freeze
    ## the first 172 layers and unfreeze the rest:
    #for layer in model.layers[:172]:
    #   layer.trainable = False
    #for layer in model.layers[172:]:
    #   layer.trainable = True

    #earlistop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
    #csv_logger = CSVLogger('training2Dense.log')
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, min_lr=1e-5)
    #callbacks_list = [earlistop,csv_logger, reduce_lr]

    #model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),loss='categorical_crossentropy', metrics =['accuracy'])

    #model.fit(trn_np, trn_labels, nb_epoch=100, batch_size=batch_size,
    #          validation_data=(val_np, val_labels), callbacks=callbacks_list)
    #model.save_weights(MODELS + "finetuning_InceptionV3.h5")

    InceptionV3_model = load_model(SaveModelName)

    ## Data augmentation
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
    plt.savefig(PATH + 'cmInceptionCropSmall.png',
                bbox_inches='tight')


    return preds





## d a global spatial average pooling layer
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
## let's add a fully-connected layer
#x = Dense(1024, activation='relu')(x)
## and a logistic layer
#predictions = Dense(4, activation='linear', name='bb')(x)
#
## this is the model we will train
#model = Model(input=base_model.input, output=predictions)
#
## first: train only the top layers (which were randomly initialized)
## i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
#    layer.trainable = False
#
#
### Number of images per gradient increment
#batch_size=200
#earlistop = EarlyStopping(monitor='loss', min_delta=0, patience=1, verbose=1, mode='auto')
#csv_logger = CSVLogger('training.log')
#reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=0, verbose=1, min_lr=1e-5)
#callbacks_list = [earlistop,csv_logger, reduce_lr]
#model.compile(Adam(lr=0.001), loss='msle', metrics=['accuracy'])
#model.fit(trn_np, trn_bbox, batch_size=batch_size, nb_epoch=50, 
#             validation_split=0.0, shuffle=True, callbacks=callbacks_list)
#
#
#print("--- Optimization  %s seconds ---" % (time.time() - start_time))
#model.save_weights(MODELS+'InceptionV3small.h5')
#
## Prediction 
#preds = model.predict(test_np, batch_size=batch_size)
#
## Save results
#boundingboxtest = np.c_[raw_test_filenames, preds, raw_test_sizes]
## Restore size of bouding box
## bb[img, 'height', 'width', 'x', 'y', 'raw width', 'raw height']
#def restore_box(i,bb):
#    conv_x = (224. / float(bb[5]))
#    conv_y = (224. / float(bb[6]))
#    bb[1] = max(float(bb[1])/conv_y, 0)
#    bb[2] = max(float(bb[2])/conv_x, 0)
#    bb[3] = max(float(bb[3])/conv_x, 0)
#    bb[4] = max(float(bb[4])/conv_y, 0)
#    return(bb)
#
#bb_test_restore = np.stack([restore_box(i, bb) for i,bb in enumerate(boundingboxtest) ])
#
## Restore bbox on test
#bbtest = pd.DataFrame(bb_test_restore)
#bbtest.to_csv(PROCESSED + 'bbox.csv', index=False)


#LoadData()
#LoadJsonLabels()
#TrainTopModel()
Pred = FineTuning()
name_classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
df = pd.DataFrame(Pred)
df.columns = name_classes
ProbabilityDistribution(df)
