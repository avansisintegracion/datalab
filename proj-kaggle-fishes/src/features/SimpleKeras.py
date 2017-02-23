from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing import image, sequence
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.np_utils import to_categorical
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np

#PATH = "../../data/interim/train/devcrop/" 
PATH = "../../data/interim/train/cropotsu/" 
MODELS = "../../models/"
PROCESSED = "../../data/processed/"

# get batches and onehot encoding label
trn = image.ImageDataGenerator().flow_from_directory(PATH + 'train', target_size=(224,224),
            class_mode='categorical', shuffle=True, batch_size=4)
trn_labels = to_categorical(trn.classes)

# get np array for data
trn_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'train', target_size=(224,224),
            class_mode=None, shuffle=False, batch_size=1)
trn_np =  np.concatenate([trn_batch.next() for i in range(trn_batch.nb_sample)])

val = image.ImageDataGenerator().flow_from_directory(PATH + 'valid', target_size=(224,224),
            class_mode='categorical', shuffle=True, batch_size=4)
val_labels = to_categorical(val.classes)

# get np array for val
val_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'valid', target_size=(224,224),
            class_mode=None, shuffle=False, batch_size=1)
val_np =  np.concatenate([val_batch.next() for i in range(val_batch.nb_sample)])

# create the base pre-trained model
#base_model = VGG16(weights='imagenet', include_top=False)
# Epoch 3/3
# 3277/3277 [==============================] - 101s - loss: 8.8042 - acc:
# 0.4538 - val_loss: 8.6393 - val_acc: 0.4640
base_model = InceptionV3(weights='imagenet', include_top=False)
# Epoch 3/3
# 3277/3277 [==============================] - 138s - loss: 0.1595 - acc:
# 0.9463 - val_loss: 0.6203 - val_acc: 0.8380
#base_model = ResNet50(weights='imagenet', include_top=False)
# Epoch 3/3
# 3277/3277 [==============================] - 176s - loss: 12.9801 - acc:
# 0.1947 - val_loss: 13.0234 - val_acc: 0.1920


# batch of images 
batch_size=64

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(8, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
batch_size=64
earlistop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')
csv_logger = CSVLogger('training.log')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=0, verbose=1, min_lr=1e-5)
callbacks_list = [earlistop,csv_logger, reduce_lr]
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trn_np, trn_labels, batch_size=batch_size, nb_epoch=50, validation_data=(val_np, val_labels), 
     callbacks=callbacks_list)

# serialize model to JSON
model_json = model.to_json()
name_json = MODELS + "model.json"
with open(name_json, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(MODELS+'SimplekerasInceptionV3small.h5')
print("Saved model to disk")

preds_val = model.predict(val_np, batch_size=batch_size)
print("Log loss in validation set: ",log_loss(val_labels, preds_val))

# Prediction 
test_batch = image.ImageDataGenerator().flow_from_directory(PATH + 'test', target_size=(224,224),
            class_mode=None, shuffle=False, batch_size=1)
test_np =  np.concatenate([test_batch.next() for i in range(test_batch.nb_sample)])
raw_test_filenames = test_batch.filenames

preds = model.predict(test_np, batch_size=batch_size)

subm = preds
# classes = sorted(batches.class_indices, key=batches.class_indices.get)
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
submission = pd.DataFrame(subm, columns=classes)
submission.insert(0, 'image', raw_test_filenames)
submission.to_csv(PROCESSED + 'classes.csv', index=False)
