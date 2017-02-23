import time
print(time.ctime()) # Current time
start_time = time.time()

# Functions and basic import libraries in the utils.py file
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image, sequence
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import pandas as pd
import math
import PIL
from PIL import Image


PATH = "../../data/interim/train/devcrop/sample/"
MODELS = "../../models/"
PROCESSED = "../../data/processed/"
ANNOS = "../../data/external/annos/"

### Load data
## Train
# Train image generator
trn_generator = image.ImageDataGenerator()
#        rotation_range=10,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True)

# Train read from batch 
trn_batch = trn_generator.flow_from_directory(PATH + 'train', target_size=(299,299),
            class_mode=None, shuffle=True, batch_size=1)
trn_np =  np.concatenate([trn_batch.next() for i in range(trn_batch.nb_sample)])
trn_labels = to_categorical(trn_batch.classes) # Classes to one-hot encoding
filenames = trn_batch.filenames # Get filenames

## Test
test_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'test', target_size=(299,299),
            class_mode=None, shuffle=False, batch_size=1)
test_np =  np.concatenate([test_batch.next() for i in range(test_batch.nb_sample)])
test_filenames = test_batch.filenames # Get filenames

print("--- Load data %s ---" % (time.time() - start_time))

### Bounding boxes & multi output
import ujson as json
anno_classes = ['alb', 'bet', 'dol', 'lag', 'shark', 'yft']

bb_json = {}
for c in anno_classes:
    j = json.load(open('{}{}_labels.json'.format(ANNOS, c), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]

# Get python raw filenames (without foldername)
raw_filenames = [f.split('/')[-1] for f in filenames]
raw_test_filenames = [f.split('/')[-1] for f in test_filenames]

# Image that have no annotation, empty bounding box
empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}
for f in raw_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox
for f in raw_test_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox
 
# The sizes of images can be related to ship sizes. Get sizes for raw image
sizes = [PIL.Image.open(PATH+'train/'+f).size for f in filenames]
raw_test_sizes = [PIL.Image.open(PATH+'test/'+f).size for f in test_filenames]

# Convert dictionary into array
bb_params = ['height', 'width', 'x', 'y']
Nbins = 20
def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (size[0] / Nbins)
    conv_y = (size[1] / Nbins)
    bb[0] = max(math.ceil((bb[0]+bb[3])/conv_y),0)
    bb[1] = max(math.ceil((bb[1]+bb[2])/conv_x),0)
    bb[2] = max(int(bb[2]/conv_x), 0)
    bb[3] = max(int(bb[3]/conv_y), 0)
    return bb

# Tranform box in terms of defined compressed image size (299)
trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, sizes)],).astype(np.float32)
trn_bbox_onehot = to_categorical(trn_bbox,nb_classes=Nbins+2)
#trn_bbox_onehot1 = to_categorical(trn_bbox[:,1],nb_classes=Nbins+2)
#trn_bbox_onehot2 = to_categorical(trn_bbox[:,2],nb_classes=Nbins+2)
#trn_bbox_onehot3 = to_categorical(trn_bbox[:,3],nb_classes=Nbins+2)

print("--- Bounding boxes reading  %s seconds ---" % (time.time() - start_time))

#InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
#                    input_tensor=None, input_shape=(299, 299, 3))
## Note that the preprocessing of InceptionV3 is:
## (x / 255 - 0.5) x 2
#
#print('Adding Average Pooling Layer and Softmax Output Layer ...')
#output = InceptionV3_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
#output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
#output = Flatten(name='flatten')(output)
#output = Dense(8, activation='softmax', name='predictions')(output)
#InceptionV3_model = Model(InceptionV3_notop.input, output)
#InceptionV3_model.summary()

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor = None)
# d a global spatial average pooling layer
x = base_model.get_layer(index = -1).output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
#x = Dense(1024, activation='relu')(x)
#x = Flatten(name='flatten')(x)
x_bb1 = Dense(Nbins+2, activation='sigmoid', name='bb1')(x)
#x_bb2 = Dense(Nbins+2, activation='sigmoid', name='bb2')(x)
#x_bb3 = Dense(Nbins+2, activation='sigmoid', name='bb3')(x)
#x_bb4 = Dense(Nbins+2, activation='sigmoid', name='bb4')(x)
#predictions = Dense(Nbins+1, activation='softmax', name='bb')(x)

# this is the model we will train
model = Model(input=base_model.input, output=[x_bb1])

## first: train only the top layers (which were randomly initialized)
## i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
#    layer.trainable = False
for layer in model.layers[:211]:
   layer.trainable = False
for layer in model.layers[211:]:
   layer.trainable = True


## Number of images per gradient increment
batch_size=64
earlistop = EarlyStopping(monitor='loss', min_delta=0, patience=1, verbose=1, mode='auto')
csv_logger = CSVLogger('training.log')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=0, verbose=1, min_lr=1e-5)
callbacks_list = [earlistop,csv_logger, reduce_lr]
model.compile(Adam(lr=0.001),loss=['categorical_crossentropy'],
    metrics=['accuracy'])
model.fit(trn_np, [trn_bbox_onehot] , batch_size=batch_size, nb_epoch=50, 
             validation_split=0.2, shuffle=True, callbacks=callbacks_list)


#print("--- FC  %s seconds ---" % (time.time() - start_time))
#model.optimizer.lr = 1e-5
#
#model.fit(trn_np, [trn_bbox_onehot0,trn_bbox_onehot1, trn_bbox_onehot2, trn_bbox_onehot3], batch_size=batch_size, nb_epoch=20, 
             #validation_split=0.2,shuffle=True, callbacks=callbacks_list)
#
#model.fit(conv_feat, trn_bbox, batch_size=batch_size, nb_epoch=80, 
#             shuffle=True)
#
print("--- Optimization  %s seconds ---" % (time.time() - start_time))
model.save_weights(MODELS+'InceptionV3small.h5')

# Prediction 
preds = model.predict(test_np, batch_size=batch_size)
bb1_label_test = [ pred.argmax() for pred in preds[0] ]
bb2_label_test = [ pred.argmax() for pred in preds[1] ]
bb3_label_test = [ pred.argmax() for pred in preds[2] ]
bb4_label_test = [ pred.argmax() for pred in preds[3] ]

# Save results
boundingboxtest = np.c_[raw_test_filenames, bb1_label_test, bb2_label_test,
        bb3_label_test, bb4_label_test, raw_test_sizes]
# Restore size of bouding box
# bb[0. img, 1. y1, 2. x2, 3. x0, 4. y0, 5.'raw width', 6.'raw height']
def restore_box(i,bb):
    conv_x = (int(bb[5]) / Nbins)
    conv_y = (int(bb[6]) / Nbins)
    bb[1] = float(bb[1])*conv_y
    bb[2] = float(bb[2])*conv_x
    bb[3] = float(bb[3])*conv_x
    bb[4] = float(bb[4])*conv_y
    return(bb)

bb_test_restore = np.stack([restore_box(i, bb) for i,bb in enumerate(boundingboxtest) ])

# Restore bbox on test
bbtest = pd.DataFrame(bb_test_restore)
#bbtest = pd.DataFrame(boundingboxtest)
bbtest.to_csv(PROCESSED + 'bbox.csv', index=False)
