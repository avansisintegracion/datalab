import time
print(time.ctime()) # Current time
start_time = time.time()

# Functions and basic import libraries in the utils.py file
from utils import *
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
import pandas as pd


PATH = "../../data/interim/train/devcrop/" 
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
trn_batch = trn_generator.flow_from_directory(PATH + 'train', target_size=(224,224),
            class_mode=None, shuffle=True, batch_size=1)
trn_np =  np.concatenate([trn_batch.next() for i in range(trn_batch.nb_sample)])
trn_labels = to_categorical(trn_batch.classes) # Classes to one-hot encoding
filenames = trn_batch.filenames # Get filenames

## Test
test_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'test', target_size=(224,224),
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
def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (224. / size[0])
    conv_y = (224. / size[1])
    bb[0] = max((bb[0]+bb[3])*conv_y, 0)
    bb[1] = max((bb[1]+bb[2])*conv_x, 0)
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb

# Tranform box in terms of defined compressed image size (224)
trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, sizes)],).astype(np.float32)

print("--- Bounding boxes reading  %s seconds ---" % (time.time() - start_time))

#base_model = vgg_ft_bn(4)
base_model = InceptionV3(weights='imagenet', include_top=False)
# d a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer
predictions = Dense(4, name='bb')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
#    layer.trainable = False
#for layer in model.layers[:211]:
#   layer.trainable = False
#for layer in model.layers[211:]:
#   layer.trainable = True



## Number of images per gradient increment
batch_size=64
earlistop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')
csv_logger = CSVLogger('training.log')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=0, verbose=1, min_lr=1e-5)
best_model_file = MODELS+'InceptionV3small.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)
callbacks_list = [earlistop,csv_logger, reduce_lr]
model.compile(Adam(lr=0.001), loss='msle', metrics=['accuracy'])
model.fit(trn_np, trn_bbox, batch_size=batch_size, nb_epoch=50, 
             validation_split=0.2, shuffle=True, callbacks=callbacks_list)


print("--- Optimization  %s seconds ---" % (time.time() - start_time))

# Prediction 
preds = model.predict(test_np, batch_size=batch_size)

# Save results
boundingboxtest = np.c_[raw_test_filenames, preds, raw_test_sizes]
# Restore size of bouding box
# bb[img, 'height', 'width', 'x', 'y', 'raw width', 'raw height']
def restore_box(i,bb):
    conv_x = (224. / float(bb[5]))
    conv_y = (224. / float(bb[6]))
    bb[1] = max(float(bb[1])/conv_y, 0)
    bb[2] = max(float(bb[2])/conv_x, 0)
    bb[3] = max(float(bb[3])/conv_x, 0)
    bb[4] = max(float(bb[4])/conv_y, 0)
    return(bb)

bb_test_restore = np.stack([restore_box(i, bb) for i,bb in enumerate(boundingboxtest) ])

# Restore bbox on test
bbtest = pd.DataFrame(bb_test_restore)
bbtest.to_csv(PROCESSED + 'bbox.csv', index=False)
