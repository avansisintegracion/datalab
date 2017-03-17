from __future__ import print_function
import glob
import os
import json
import cv2
import numpy as np
import shutil
import pickle
import os.path as op

import ipdb

from src.data import DataModel as dm

class CropBoundingBoxes(object):
    '''Croppping images, relabel, and train val separation'''
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

    def process_annos(self, label_file):
        file_name = os.path.basename(label_file)
        class_name = file_name.split("_")[0]
        if not os.path.isdir(op.join(self.f.data_interim_train_crop_train,class_name.upper())):
            os.makedirs(op.join(self.f.data_interim_train_crop_train, class_name.upper()))
        print("Cropping ", class_name)
        with open(label_file) as data_file:
            data = json.load(data_file)

        for img_data in data:
            # Shark and yft have different 'filename'
            if class_name == "shark" or class_name == "yft" or class_name == "OTHER": 
                img_file = op.join(self.f.data_raw_train, class_name.upper(), img_data['filename'].split("/")[-1])
            else: img_file = op.join(self.f.data_raw_train, class_name.upper(), img_data['filename'])
            img = cv2.imread(img_file)
            # Crop only images with both heads and tails present for cleaner dataset
            if len(img_data['annotations']) >= 1:
                x0 = max(int(img_data['annotations'][0]['x']), 0)
                y0 = max(int(img_data['annotations'][0]['y']), 0)
                x1 = max(int(img_data['annotations'][0]['x'] + img_data['annotations'][0]['width']), 0)
                y1 = max(int(img_data['annotations'][0]['y'] + img_data['annotations'][0]['height']), 0)
                diff = (x1 - x0)  - (y1 - y0)
                if diff > 0:
                    x_left = x0
                    x_right = x1
                    y_up = max(y0 - diff/2,0)
                    y_down = y1 + diff/2
                else:
                    x_left = max(x0 - abs(diff)/2,0)
                    x_right = x1 + abs(diff)/2
                    y_up = y0
                    y_down = y1

                #print(img_file, x_left,x_right,y_up,y_down,diff )
                img = img[y_up:y_down, x_left:x_right, :]
                #img = img[y0:y1, x0:x1, :]
                if class_name == "shark" or class_name == "yft" or class_name == "OTHER": 
                    cv2.imwrite(op.join(self.f.data_interim_train_crop_train, class_name.upper(),
                            img_data['filename'].split("/")[-1]), img)
                else: cv2.imwrite(op.join(self.f.data_interim_train_crop_train, class_name.upper(),
                    img_data['filename']), img)
            else:
                if class_name == "shark" or class_name == "yft" or class_name == "OTHER":
                    shutil.copyfile(op.join(self.f.data_raw_train, class_name.upper(), 
                        img_data['filename'].split("/")[-1]),
                            op.join(self.f.data_interim_train_crop_train, class_name.upper(),
                            img_data['filename'].split("/")[-1]))
                else:
                    shutil.copyfile(op.join(self.f.data_raw_train, class_name.upper(),
                        img_data['filename']), op.join(self.f.data_interim_train_crop_train,
                            class_name.upper(), img_data['filename']))

    def make_cropped_dataset(self):
        label_files = glob.glob(op.join(self.f.data_external_annos, '*.json' ))
        for file in label_files:
            self.process_annos(file)

    def copy_nofish(self):
        if not os.path.isdir(op.join(self.f.data_interim_train_crop_train, "NoF")):
            print("Copy whole NoF folder")
            shutil.copytree(op.join(self.f.data_raw_train, "NoF"),
                    op.join(self.f.data_interim_train_crop_train, 'NoF'))

    def relabel(self):
        print("Modifiying wrong labels from train set")
        relabelsfile = op.join(self.f.data_external, "relabels/relabels.csv")
        with open(relabelsfile) as f:
            counter = 0
            for line in f:
                cols = line.split()
                src = op.join(self.f.data_interim_train_crop_train, cols[1], cols[0]+".jpg")
                dst = op.join(self.f.data_interim_train_crop_train, cols[2], cols[0]+".jpg")
                try:
                    os.rename(src, dst)
                    counter+=1
                    #print("{} ".format(src))

                except:
                    pass
                    #print("{} not found".format(src))

        print("---> Modified {} photos".format(counter))

    def separate_val(self):
        print('Separating validation from train')
        for k, img in self.training_img.iteritems():
            if img['validation'] is True:
                fromname = op.join(self.f.data_interim_train_crop_train, img['fishtype'])
                toname = op.join(self.f.data_interim_train_crop_val, img['fishtype'])
                if not os.path.isdir(toname):
                    os.makedirs(toname)
                try: 
                    os.rename(op.join(fromname, img['imgname']),
                            op.join(toname, img['imgname']))
                except:
                    pass
                    #print("Separation not found for", img['imgname'])

    def process_multiple_crop(self, label_file):
        file_name = os.path.basename(label_file)
        class_name = file_name.split("_")[0]
        if not os.path.isdir(op.join(self.f.data_interim_train_crop_train,class_name.upper())):
            os.makedirs(op.join(self.f.data_interim_train_crop_train, class_name.upper()))
        print("Adding multiple image crops in ", class_name)
        with open(label_file) as data_file:
            data = json.load(data_file)

        for img_data in data:
            # Shark and yft have different 'filename'
            if class_name == "shark" or class_name == "yft" or class_name == "OTHER": 
                img_file = op.join(self.f.data_raw_train, class_name.upper(), img_data['filename'].split("/")[-1])
            else: img_file = op.join(self.f.data_raw_train, class_name.upper(), img_data['filename'])
            img = cv2.imread(img_file)
            # Crop only images with both heads and tails present for cleaner dataset
            if len(img_data['annotations']) >= 1:
                for idx, val in enumerate(img_data['annotations']):
                    if(idx == 0): continue
                    x0 = max(int(img_data['annotations'][0]['x']), 0)
                    y0 = max(int(img_data['annotations'][0]['y']), 0)
                    x1 = max(int(img_data['annotations'][0]['x'] + img_data['annotations'][0]['width']), 0)
                    y1 = max(int(img_data['annotations'][0]['y'] + img_data['annotations'][0]['height']), 0)
                    diff = (x1 - x0)  - (y1 - y0)
                    if diff > 0:
                        x_left = x0
                        x_right = x1
                        y_up = max(y0 - diff/2,0)
                        y_down = y1 + diff/2
                    else:
                        x_left = max(x0 - abs(diff)/2,0)
                        x_right = x1 + abs(diff)/2
                        y_up = y0
                        y_down = y1

                    #print(img_file, x_left,x_right,y_up,y_down,diff )
                    imgC = img[y_up:y_down, x_left:x_right, :]
                    #img = img[y0:y1, x0:x1, :]
                    if class_name == "shark" or class_name == "yft" or class_name == "OTHER": 
                        cv2.imwrite(op.join(self.f.data_interim_train_crop_train, class_name.upper(),
                                str(idx) + img_data['filename'].split("/")[-1]), imgC)
                    else: cv2.imwrite(op.join(self.f.data_interim_train_crop_train, class_name.upper(),
                        str(idx) + img_data['filename']), imgC)
            else:
                if class_name == "shark" or class_name == "yft" or class_name == "OTHER":
                    shutil.copyfile(op.join(self.f.data_raw_train, class_name.upper(), 
                        img_data['filename'].split("/")[-1]),
                            op.join(self.f.data_interim_train_crop_train, class_name.upper(),
                            img_data['filename'].split("/")[-1]))
                else:
                    shutil.copyfile(op.join(self.f.data_raw_train, class_name.upper(),
                        img_data['filename']), op.join(self.f.data_interim_train_crop_train,
                            class_name.upper(), img_data['filename']))

    def crop_multiple_fish(self):
        label_files = glob.glob(op.join(self.f.data_external_annos, '*.json' ))
        for file in label_files:
            self.process_multiple_crop(file)

#    def main(self):
#        print("main")
#        self.make_cropped_dataset()
#        self.copy_nofish()
#        self.relabel()
#        self.separate_val()
#        self.crop_multiple_fish()

if __name__ == '__main__':
    os.chdir(op.dirname(op.abspath(__file__)))
    #CropBoundingBoxes().main()
    CropBoundingBoxes().make_cropped_dataset()
    CropBoundingBoxes().copy_nofish()
    CropBoundingBoxes().relabel()
    CropBoundingBoxes().separate_val()
    CropBoundingBoxes().crop_multiple_fish()
