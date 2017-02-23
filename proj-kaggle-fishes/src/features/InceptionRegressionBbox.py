import time
print(time.ctime()) # Current time
start_time = time.time()

# Functions and basic import libraries in the utils.py file
from utils import *
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
import pandas as pd


PATH = "../../data/interim/train/devcrop/" 
MODELS = "../../models/"
PROCESSED = "../../data/processed/"
ANNOS = "../../data/external/annos/"

# Train set
trn_generator = image.ImageDataGenerator(
        rotation_range=10,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

trn_batch = trn_generator.flow_from_directory(PATH + 'train', target_size=(360,640),
            class_mode=None, shuffle=False, batch_size=1)
trn_labels = to_categorical(trn_batch.classes)
trn_np =  np.concatenate([trn_batch.next() for i in range(trn_batch.nb_sample)])


# Test set
test_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'test', target_size=(360,640),
            class_mode=None, shuffle=False, batch_size=1)
test_np =  np.concatenate([test_batch.next() for i in range(test_batch.nb_sample)])

# Get filenames
filenames = trn_batch.filenames
test_filenames = test_batch.filenames

## Bounding boxes & multi output
import ujson as json
anno_classes = ['alb', 'bet', 'dol', 'lag', 'shark', 'yft']

bb_json = {}
for c in anno_classes:
    j = json.load(open('{}{}_labels.json'.format(ANNOS, c), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]

# Get python raw filenames
raw_filenames = [f.split('/')[-1] for f in filenames]
raw_test_filenames = [f.split('/')[-1] for f in test_filenames]

# Get file index
file2idx = {o:i for i,o in enumerate(raw_filenames)}
test_file2idx = {o:i for i,o in enumerate(raw_test_filenames)}

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
    conv_x = (640. / size[0])
    conv_y = (360. / size[1])
    bb[0] = max((bb[0]+bb[3])*conv_y, 0)
    bb[1] = max((bb[1]+bb[2])*conv_x, 0)
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb

# Tranform box in terms of defined compressed image size (224)
trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, sizes)],).astype(np.float32)

print("--- Annotations  %s seconds ---" % (time.time() - start_time))

base_model = vgg_ft_bn(4)

#base_model.compile(optimizer=Adam(1e-3),
#       loss='msle', metrics=['accuracy'])
#
## Number of images per gradient increment
batch_size=64
#base_model.fit(trn_np, trn_bbox, batch_size=batch_size, nb_epoch=3,
#       validation_split=0.2)
#
#base_model.optimizer.lr = 1e-5
#base_model.fit(trn_np, trn_bbox, batch_size=batch_size, nb_epoch=10,
#       validation_split=0.2)

conv_layers,fc_layers = split_at(base_model, Convolution2D)
conv_model = Sequential(conv_layers)

conv_feat = conv_model.predict(trn_np)
conv_test_feat = conv_model.predict(test_np)

print("--- Feature extraction  %s seconds ---" % (time.time() - start_time))

# Two outputs: class and bounding boxes
p=0.5
inp = Input(conv_layers[-1].output_shape[1:])
#inp = base_model.output
x = MaxPooling2D()(inp)
x = BatchNormalization(axis=1)(x)
x = Dropout(p)(x)
#x = Dropout(p/4)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
#x = Dropout(p)(x)
x = Dropout(p/2)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
#x = Dropout(p/2)(x)
x = Dropout(p/4)(x)
x_bb = Dense(4, activation='linear', name='bb')(x)
#x_class = Dense(8, activation='softmax', name='class')(x)

model = Model([inp], [x_bb])
#model = Model(input=base_model.input, output=x_bb)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
#    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(Adam(lr=0.001), loss='msle', metrics=['accuracy'])
model.fit(conv_feat, trn_bbox, batch_size=batch_size, nb_epoch=3, 
             validation_split=0.2, shuffle=True)

print("--- FC  %s seconds ---" % (time.time() - start_time))
model.optimizer.lr = 1e-5

model.fit(conv_feat, trn_bbox, batch_size=batch_size, nb_epoch=5, 
             validation_split=0.2, shuffle=True)


print("--- Optimization  %s seconds ---" % (time.time() - start_time))
model.save_weights(MODELS+'InceptionV3small.h5')

# Prediction 
preds = model.predict(conv_test_feat, batch_size=batch_size)

# Save results
boundingboxtest = np.c_[raw_test_filenames, preds, raw_test_sizes]
# Restore size of bouding box
# bb[img, 'height', 'width', 'x', 'y', 'raw width', 'raw height']
def restore_box(i,bb):
    conv_x = (640. / float(bb[5]))
    conv_y = (360. / float(bb[6]))
    bb[1] = max(float(bb[1])/conv_y, 0)
    bb[2] = max(float(bb[2])/conv_x, 0)
    bb[3] = max(float(bb[3])/conv_x, 0)
    bb[4] = max(float(bb[4])/conv_y, 0)
    return(bb)

bb_test_restore = np.stack([restore_box(i, bb) for i,bb in enumerate(boundingboxtest) ])

# Restore bbox on test
bbtest = pd.DataFrame(bb_test_restore)
bbtest.to_csv(PROCESSED + 'bbox.csv', index=False)
