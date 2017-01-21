import time
print(time.ctime()) # Current time
start_time = time.time()

# Functions and basic import libraries in the utils.py file
from utils import *

# 'VGG', which won the 2014 Imagenet competition, and is a very simple model to
# create and understand. The VGG Imagenet team created both a larger, slower,
# slightly more accurate model (VGG 19) and a smaller, faster model (VGG 16).
# We will be using VGG 16 since the much slower performance of VGG19 is
# generally not worth the very minor improvement in accuracy.
from vgg16bn import Vgg16BN

PATH = "../../data/interim/"
MODELS = "../../models/"
LABELS = "../../data/external/labels/"
ANNOS = "../../data/external/annos/"

# batch of images 
batch_size=64

# Get batches for training and validation
batches = get_batches(PATH+'train', batch_size=batch_size)
val_batches = get_batches(PATH+'valid', batch_size=batch_size*2, shuffle=False)

# Get classes for val, trn, label
(val_classes, trn_classes, val_labels, trn_labels, 
    val_filenames, filenames, test_filenames) = get_classes(PATH)


# Get the data into a variable
trn = get_data(PATH+'train')
val = get_data(PATH+'valid')
test = get_data(PATH+'test')

# The model has 8 classes to predict
model = vgg_ft_bn(8)
model.compile(optimizer=Adam(1e-3),
       loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(trn, trn_labels, batch_size=batch_size, nb_epoch=3, validation_data=(val, val_labels))
model.save_weights(MODELS+'ft1.h5')
print("--- %s seconds ---" % (time.time() - start_time))

### Precompute convolutional outpu
# The last convulotional layer of VGG is unlikely to change 

conv_layers,fc_layers = split_at(model, Convolution2D)
conv_model = Sequential(conv_layers)

conv_feat = conv_model.predict(trn)
conv_val_feat = conv_model.predict(val)
conv_test_feat = conv_model.predict(test)

print("--- Loading  %s seconds ---" % (time.time() - start_time))

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
raw_val_filenames = [f.split('/')[-1] for f in val_filenames]

# Get file index
file2idx = {o:i for i,o in enumerate(raw_filenames)}
val_file2idx = {o:i for i,o in enumerate(raw_val_filenames)}

# Image that have no annotation, empty bounding box
empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}

for f in raw_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox
for f in raw_val_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox
 
           
# The sizes of images can be related to ship sizes. Get sizes for raw image
sizes = [PIL.Image.open(PATH+'train/'+f).size for f in filenames]
raw_val_sizes = [PIL.Image.open(PATH+'valid/'+f).size for f in val_filenames]

# Convert dictionary into array
bb_params = ['height', 'width', 'x', 'y']
def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (224. / size[0])
    conv_y = (224. / size[1])
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb

# Tranform box in terms of defined compressed image size (224)
trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, sizes)],).astype(np.float32)
val_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_val_filenames, raw_val_sizes)]).astype(np.float32)

print("--- Annotations  %s seconds ---" % (time.time() - start_time))

# Two outputs: class and bounding boxes
p=0.6

inp = Input(conv_layers[-1].output_shape[1:])
x = MaxPooling2D()(inp)
x = BatchNormalization(axis=1)(x)
x = Dropout(p/4)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p/2)(x)
x_bb = Dense(4, name='bb')(x)
x_class = Dense(8, activation='softmax', name='class')(x)

# Multiple outputs, then we need to provide them to the model constructor in an array, 
# and we also need to say what loss function to use for each. We also weight 
# the bounding box loss function down by 1000x since the scale of the cross-entropy loss 
# and the MSE is very different.
model = Model([inp], [x_bb, x_class])
model.compile(Adam(lr=0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'],
             loss_weights=[.001, 1.])

model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=batch_size, nb_epoch=3, 
             validation_data=(conv_val_feat, [val_bbox, val_labels]))

print("--- FC  %s seconds ---" % (time.time() - start_time))
model.optimizer.lr = 1e-5

model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=batch_size, nb_epoch=10, 
             validation_data=(conv_val_feat, [val_bbox, val_labels]))

print("--- Optimization  %s seconds ---" % (time.time() - start_time))
model.save_weights(MODELS+'bn_anno.h5')

# Prediction 
preds = model.predict(conv_test_feat, batch_size=batch_size)

boundinboxtest = preds[0]
bbtest = pd.DataFrame(boundinboxtest)
bbtest.insert(0, 'image', raw_test_filenames)
bbtest.to_csv(MODELS + 'bbox.csv', index=False)

#save_array(MODELS + 'bbox.dat', boundinboxtest)

subm = preds[1]
# classes = sorted(batches.class_indices, key=batches.class_indices.get)
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
submission = pd.DataFrame(subm, columns=classes)
submission.insert(0, 'image', raw_test_filenames)
submission.to_csv(MODELS + 'classes.csv', index=False)
