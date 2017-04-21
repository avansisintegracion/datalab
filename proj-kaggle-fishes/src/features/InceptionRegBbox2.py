import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import os
import os.path as op
import glob
import PIL
import time
import ujson as json
import ipdb
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.preprocessing import image
from keras.callbacks import (EarlyStopping, CSVLogger, ReduceLROnPlateau,
                             ModelCheckpoint)
from keras.optimizers import SGD
from src.data import DataModel as dm
from keras.utils.np_utils import to_categorical
from keras import backend as K
print(time.ctime())  # Current time
start_time = time.time()
K.set_image_dim_ordering('tf')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PATH = "../../data/interim/train/devcrop/" 
MODELS = "../../models/"
seed = 7
np.random.seed(seed)


class InceptionFineTuning(object):
    '''Using keras with pretrained InceptionV3'''
    def __init__(self):
        self.f = dm.ProjFolder()
        self.classes = ['ALB',
                        'BET',
                        'DOL',
                        'LAG',
                        'NoF',
                        'OTHER',
                        'SHARK',
                        'YFT']
        with open(op.join(self.f.data_processed, 'training_images.json'), 'rb') as file:
            self.training_img = json.load(file)
        with open(op.join(self.f.data_processed, 'test_images.json'), 'rb') as file:
            self.test_img = json.load(file)

    def TrainModel(self, train_generator, validation_generator, img_height,
                   img_width, nbr_train_samples, nbr_val_samples):
        learning_rate = 1e-3
        nbr_epoch = 50
        print('Loading InceptionV3 Weights ...')
        base_model = InceptionV3(include_top=False, weights='imagenet',
                                 input_tensor=None, input_shape=(img_height,
                                                                 img_width, 3))

        print("--- Adding on top layers %.1f seconds ---" % (time.time() -
                                                             start_time))
        output = base_model.get_layer(index=-1).output  # Shape: (8, 8, 2048)
        output = AveragePooling2D((8, 8), strides=(8, 8),
                                  name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        x_bb = Dense(4, name='bb')(output)
        x_fish = Dense(1, activation='sigmoid', name='fish')(output)

        #model = Model(base_model.input, x_bb)
        #model = Model(base_model.input, x_fish)
        model = Model(base_model.input, [x_bb, x_fish])

        optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.0,
                        nesterov=True)
        model.compile(loss=['mse', 'binary_crossentropy'],
                      optimizer=optimizer, metrics=['accuracy'],
                      loss_weights=[0.001, 1.])
                      #loss_weights=[.00001, 1.])
        #model.compile(loss='mse', optimizer=optimizer,
        #              metrics=['accuracy'], loss_weights=[.001])
        #model.compile(loss='binary_crossentropy', optimizer=optimizer,
        #              metrics=['accuracy'], loss_weights=[1.])
        # print(model.summary())

        earlistop = EarlyStopping(monitor='val_bb_acc', min_delta=0, patience=0,
                                  verbose=1, mode='auto')
        csv_logger = CSVLogger(op.join(self.f.data_interim_train_devcrop,
                                       'trainingInceptionCropSmall.log'))
        reduce_lr = ReduceLROnPlateau(monitor='val_bb_loss', factor=0.1,
                                      patience=1, verbose=1, min_lr=1e-5)
        SaveModelName = MODELS + "InceptionV3BboxFish.h5"
        best_model = ModelCheckpoint(SaveModelName, monitor='val_bb_acc',
                                     verbose=1, save_best_only=True)
        callbacks_list = [earlistop, csv_logger, reduce_lr, best_model]

        # Testing bbox figure ----------------

        def create_rect(bb, color='red'):
            # height = bb[0]
            # width = bb[1]
            # x = bb[2]
            # y = bb[3]
            # return plt.Rectangle((x, y), width, height, color=color, fill=False,
            # lw=1)
            x0 = bb[0]
            y0 = bb[1]
            width = abs(bb[2] - bb[0])
            height = abs(bb[3] - bb[1])
            return plt.Rectangle((x0, y0), width, height, color=color, fill=False,
                                 lw=1)

        #fig, ax = plt.subplots(2, 4, sharex=False, sharey=False)
        #for i in range(8):
        #    img_bb = validation_generator.next()
        #    ax[i/4, i % 4].imshow(img_bb[0][i], interpolation='nearest')
        #    ax[i/4, i % 4].axis('off')
        #    ax[i/4, i % 4].add_patch(create_rect(img_bb[1][0][i]))
        #    fig.subplots_adjust(hspace=0, wspace=0)

        #plt.savefig("generator_photos.jpg")

        #ipdb.set_trace()

        # Testing bbox figure ----------------


        model.fit_generator(
                train_generator,
                samples_per_epoch=nbr_train_samples,
                nb_epoch=nbr_epoch,
                validation_data=validation_generator,
                nb_val_samples=nbr_val_samples,
                callbacks=callbacks_list)

        print("--- Training model %.1f seconds ---" % (time.time() -
                                                       start_time))
        return model, SaveModelName

    def Predictions(self, model, img_height, img_width, batch_size, SaveModelName):
        print("--- Starting prediction %.1f seconds ---" % (time.time() -
                                                            start_time))
        nbr_test_samples = len(self.test_img)

        # Data augmentation for prediction
        nbr_augmentation = 5
        test_datagen = image.ImageDataGenerator(rescale=1./255)

        for idx in range(nbr_augmentation):
            print('{}th augmentation for testing ...'.format(idx))
            random_seed = np.random.random_integers(0, 100000)

            test_generator = test_datagen.flow_from_directory(
                    op.join(self.f.data_interim_train_devcrop_test),
                    target_size=(img_height, img_width),
                    batch_size=batch_size,
                    shuffle=False,  # Important !!!
                    seed=random_seed,
                    # classes = None,
                    class_mode=None)

            print('Begin to predict for testing data ...')
            if idx == 0:
                preds = model.predict_generator(test_generator,
                                                nbr_test_samples)
            else:
                preds[0] += model.predict_generator(test_generator,
                                                 nbr_test_samples)[0]
                preds[1] += model.predict_generator(test_generator,
                                                 nbr_test_samples)[1]

        preds_bb = preds[0]/nbr_augmentation
        preds_fish = preds[1]/nbr_augmentation

        def restore_box(bb, size):
            conv_x = (float(size[0]) / float(img_width))
            conv_y = (float(size[1]) / float(img_height))
            #bb[0] = float(bb[0])*conv_y
            #bb[1] = float(bb[1])*conv_x
            #bb[2] = float(bb[2])*conv_x
            #bb[3] = float(bb[3])*conv_y
            bb[0] = float(bb[0])*conv_x
            bb[1] = float(bb[1])*conv_y
            bb[2] = float(bb[2])*conv_x
            bb[3] = float(bb[3])*conv_y
            return(bb)

        test_filenames = test_generator.filenames
        raw_test_filenames = [f.split('/')[-1] for f in test_filenames]
        raw_test_sizes = [PIL.Image.open(op.join(self.f.data_raw_test, f)).size for f in raw_test_filenames]
        bb_test_restore = np.stack([restore_box(bb, s) for bb, s in zip(preds_bb, raw_test_sizes)])

        # Restore bbox on test
        bbtest = pd.DataFrame(np.concatenate((bb_test_restore, preds_fish),
                                             axis=1), index=raw_test_filenames)
        bbtest.to_csv(op.join(self.f.data_processed, 'bbox.csv'))
        print("--- End %.1f seconds ---" % (time.time() - start_time))

    def Process(self):
        img_width = 640
        img_height = 360
        batch_size = 8
        nbr_val_samples = len(glob.glob(PATH + 'val/*/*.jpg')) #/batch_size*batch_size
        # nbr_val_samples = sum(1 for k in self.training_img.values() if k.get('validation'))
        nbr_train_samples = len(glob.glob(PATH + 'train/*/*.jpg')) #/batch_size*batch_size
        # nbr_train_samples =  len(self.training_img) - nbr_val_samples

        print("Parametres: img_width {}, batch_size {}, number of train {},\
              number of val {}".format(img_width, batch_size,
              nbr_train_samples, nbr_val_samples))

        # Transformation for train
        train_datagen = image.ImageDataGenerator(rescale=1./255,)

        trn_generator = train_datagen.flow_from_directory(
                PATH + 'train',
                target_size=(img_height, img_width),
                batch_size=batch_size,
                shuffle=False,
                #save_to_dir = PATH + 'TransfTrain/',
                #save_prefix = 'aug',
                #classes = self.classes,
                class_mode=None,
                seed=seed)

        # Transformation for validation set
        val_datagen = image.ImageDataGenerator(rescale=1./255)

        val_generator = val_datagen.flow_from_directory(
            PATH + 'val',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False,
            #save_to_dir = PATH + 'TransfVal/',
            #save_prefix = 'aug',
            #classes = self.classes,
            class_mode=None,
            seed=seed)

        filenames = trn_generator.filenames
        val_filenames = val_generator.filenames

        # Bounding boxes & multi output
        anno_classes = glob.glob(op.join(self.f.data_external_annos, '*.json'))

        bb_json = {}
        for fish_class in anno_classes:
            fish_bb_json = json.load(open(op.join(fish_class), 'r'))
            for fish_annotation in fish_bb_json:
                if 'annotations' in fish_annotation.keys() and len(fish_annotation['annotations']) > 0:
                    bb_json[fish_annotation['filename'].split('/')[-1]] = { 
                        'x_0' : sorted(fish_annotation['annotations'], key=lambda x:
                        x['x'])[0]['x'], 
                        'y_0' : sorted(fish_annotation['annotations'], key=lambda x:
                        x['y'])[0]['y'], 
                        'x_1' : (sorted(fish_annotation['annotations'], key=lambda x:
                        x['x']+x['width'])[-1]['x'] + sorted(fish_annotation['annotations'], key=lambda x:
                        x['x']+x['width'])[-1]['width']),
                        'y_1' : (sorted(fish_annotation['annotations'], key=lambda x:
                        x['y']+x['height'])[-1]['y'] + sorted(fish_annotation['annotations'], key=lambda x:
                        x['y']+x['height'])[-1]['width'])
                    }
                    #if len(fish_annotation['annotations']) > 2:
                    #bb_json[fish_annotation['filename'].split('/')[-1]] = sorted(
                    #    fish_annotation['annotations'], key=lambda x:
                    #    x['height']*x['width'])[-1]

        #ipdb.set_trace()

        # Get python raw filenames
        raw_filenames = [f.split('/')[-1] for f in filenames]
        raw_val_filenames = [f.split('/')[-1] for f in val_filenames]

        # Image that have no annotation, empty bounding box
        empty_bbox = {'x_0': 0., 'y_0': 0., 'x_1': 0., 'y_1': 0.}

        for f in raw_filenames:
            if not f in bb_json.keys(): bb_json[f] = empty_bbox

        for f in raw_val_filenames:
            if not f in bb_json.keys(): bb_json[f] = empty_bbox

        # Get sizes for raw image
        sizes = [PIL.Image.open(PATH+'train/'+f).size for f in filenames]
        raw_val_sizes = [PIL.Image.open(PATH+'val/'+f).size for f in
                         val_filenames]

        counter = 0
        for a in bb_json.values():
            if float(a['x_1'] - a['x_0']) < 1.:
                counter += 1
        print('Number of empty box: ', counter)

        # Convert dictionary into array
        #               0       1       2    3
        #bb_params = ['height', 'width', 'x', 'y']
        bb_params = ['x_0', 'y_0', 'x_1', 'y_1']

        def convert_bb(bb, size):
            bb = [bb[p] for p in bb_params]
            conv_x = (img_width / float(size[0]))
            conv_y = (img_height / float(size[1]))
            #bb[0] = max((bb[0]+bb[3])*conv_y, 0)  # y1
            #bb[1] = max((bb[1]+bb[2])*conv_x, 0)  # x1
            #bb[2] = max(bb[2]*conv_x, 0)  # x0
            #bb[3] = max(bb[3]*conv_y, 0)  # y0
            bb[0] = bb[0]*conv_x # x_0
            bb[1] = bb[1]*conv_y # y_0
            bb[2] = max(bb[2]*conv_x, 0) # x_1
            bb[3] = max(bb[3]*conv_y, 0) # y_1
            # ipdb.set_trace()
            return bb

        # Tranform box in terms of defined compressed image size (224)
        trn_bbox = np.stack([convert_bb(bb_json[f], s) for f, s in
                             zip(raw_filenames, sizes)],).astype(np.float32)
        val_bbox = np.stack([convert_bb(bb_json[f], s) for f, s in
                             zip(raw_val_filenames,
                                 raw_val_sizes)],).astype(np.float32)

        # 4NoFish and 5Other
        trn_fish_labels = np.asarray([ 0 if ( fish == 4 ) else 1 for fish in trn_generator.classes ])
        val_fish_labels = np.asarray([ 0 if ( fish == 4 ) else 1 for fish in val_generator.classes ])

        #trn_fish_generator = (n for n in itertools.cycle(batch(trn_fish_labels,
        #                                                       n=batch_size)))
        #val_fish_generator = (n for n in itertools.cycle(batch(val_fish_labels,
        #                                                       n=batch_size)))

        def batch(iterable1, iterable2, n=1):
            l1 = len(iterable1)
            l2 = len(iterable2)
            for ndx in range(0, l2, n):
                #yield iterable1[ndx:min(ndx + n, l1)]
                yield [iterable1[ndx:min(ndx + n, l1)], iterable2[ndx:min(ndx + n, l2)]]

        trn_bbox_generator = (n for n in itertools.cycle(batch(trn_bbox, trn_fish_labels,
                                                               n=batch_size)))
        val_bbox_generator = (n for n in itertools.cycle(batch(val_bbox, val_fish_labels,
                                                               n=batch_size)))


        train_generator = itertools.izip(trn_generator, trn_bbox_generator)
        validation_generator = itertools.izip(val_generator, val_bbox_generator)
        #ipdb.set_trace()

        model, SaveModelName = self.TrainModel(train_generator,
                                               validation_generator,
                                               img_height, img_width,
                                               nbr_train_samples,
                                               nbr_val_samples)
        model = load_model(SaveModelName)

        print("Evualuate save model: log loss and accuarcy: \n ",
              model.evaluate_generator(validation_generator, nbr_val_samples))

        self.Predictions(model, img_height, img_width, batch_size,
                        SaveModelName)

if __name__ == '__main__':
    os.chdir(op.dirname(op.abspath(__file__)))
    InceptionFineTuning().Process()
