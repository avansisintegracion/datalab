from skimage.data import imread
from skimage.io import imshow,imsave
from skimage import img_as_float
import pandas as pd
import numpy as np
import cv2
from skimage.util import crop
from skimage.transform import rotate
from skimage.transform import resize
import matplotlib.pyplot as plt
import math
import json
import os.path as op
import os

from src.data import DataModel as dm

# os.chdir(op.dirname(op.abspath(__file__)))
os.chdir('/Users/mkoutero/Documents/Github/datalab/proj-kaggle-fishes/src/data')

def deg_angle_between(x1,y1,x2,y2):
    from math import atan2, degrees, pi
    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = degrees(rads)
    return(degs)


def get_rotated_cropped_fish(img,x1,y1,x2,y2):
    (h,w) = img.shape[:2]
    #calculate center and angle
    center = ( (x1+x2) / 2,(y1+y2) / 2)
    angle = np.floor(-deg_angle_between(x1,y1,x2,y2))
    #print('angle=' +str(angle) + ' ')
    #print('center=' +str(center))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    fish_length = np.sqrt((x1-x2)**2+(y1-y2)**2)
    cropped = rotated[(max((center[1]-fish_length/1.8),0)):(max((center[1]+fish_length/1.8),0)) ,
                      (max((center[0]- fish_length/1.8),0)):(max((center[0]+fish_length/1.8),0))]
    #imshow(img)
    #imshow(rotated)
    #imshow(cropped)
    resized = resize(cropped,(224,224))
    return(resized)


f = dm.ProjFolder()
with open(op.join(f.data_processed, 'training_images.json'), 'rb') as file:
    training_img = json.load(file)

ROOT = f.data_external_rotate_crop
label_files = {'BET': op.join(ROOT, 'BET/bet_labels.json'),
               'ALB': op.join(ROOT, 'ALB/alb_labels.json'),
               'YFT': op.join(ROOT, 'YFT/yft_labels.json'),
               'DOL': op.join(ROOT, 'DOL/dol_labels.json'),
               'SHARK': op.join(ROOT, 'SHARK/shark_labels.json'),
               'LAG': op.join(ROOT, 'LAG/lag_labels.json'),
               'OTHER': op.join(ROOT, 'OTHER/other_labels.json')}

for k, im in training_img.iteritems():
    try:
        if im['fishtype'] in ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']:
            labels = pd.read_json(label_files[im['fishtype']])
            l1 = pd.DataFrame((labels[labels.filename==im['imgname']].annotations).iloc[0])
            image = imread(op.join(im['imgpath'], im['imgname']))
            rotimage = get_rotated_cropped_fish(image, np.floor(l1.iloc[0,1]), np.floor(l1.iloc[0,2]), np.floor(l1.iloc[1,1]), np.floor(l1.iloc[1,2]))
        else:
            image = imread(op.join(im['imgpath'], im['imgname']))
            rotimage = resize(image, (224, 224))
        if im['validation'] is False:
            if not os.path.isdir(op.join(f.data_interim_train_rotatecrop_train, im['fishtype'])):
                os.mkdir(op.join(f.data_interim_train_rotatecrop_train, im['fishtype']))
            imsave(op.join(f.data_interim_train_rotatecrop_train, im['fishtype'], im['imgname']), rotimage)
        else:
            if not os.path.isdir(op.join(f.data_interim_train_rotatecrop_val, im['fishtype'])):
                os.mkdir(op.join(f.data_interim_train_rotatecrop_val, im['fishtype']))
            imsave(op.join(f.data_interim_train_rotatecrop_val, im['fishtype'], im['imgname']), rotimage)

    except:
        pass
