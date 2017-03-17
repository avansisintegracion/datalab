from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing import image, sequence
from keras.utils.np_utils import to_categorical
import numpy as np


PATH = "../../data/interim/train/devcrop/" 
#PATH = "../../data/interim/train/crop/" 

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
#base_model = InceptionV3(weights='imagenet', include_top=False)
# Epoch 3/3
# 3277/3277 [==============================] - 138s - loss: 0.1595 - acc:
# 0.9463 - val_loss: 0.6203 - val_acc: 0.8380
base_model = ResNet50(weights='imagenet', include_top=False)
# Epoch 3/3
# 3277/3277 [==============================] - 176s - loss: 12.9801 - acc:
# 0.1947 - val_loss: 13.0234 - val_acc: 0.1920

# batch of images 
batch_size=64

#model = VGG16(weights='imagenet', include_top=False)
#img_path = 'elephant.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

trn_features = base_model.predict(trn_np)
val_features = base_model.predict(val_np)

#model.fit(trn_np, trn_labels, batch_size=batch_size, nb_epoch=3, validation_data=(val_np, val_labels))
