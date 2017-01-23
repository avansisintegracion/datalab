<<<<<<< HEAD
%%time
import mxnet as mx
import xgboost as xgb
=======
import mxnet as mx
#import xgboost as xgb
>>>>>>> origin/cut-fish
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool
import os
from sklearn import cross_validation
import joblib
import pickle
<<<<<<< HEAD
import glob2
=======
import glob
>>>>>>> origin/cut-fish

def preprocessing_image(img):
    img = cv2.imread(img)
    # Convert opencv BGR to RGB
    img = img[:, :, [2, 1, 0]]
    # Make images square
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    img = img[yy: yy + short_egde, xx: xx + short_egde]
    # Resize images
    img = cv2.resize(img, (500, 500))
    return img.flatten()

<<<<<<< HEAD
# Data expected to be in : ./data/train
# Script expected to be in ./scripts/data
data = pd.DataFrame(columns=('FishType', 'ImgMatrix'))
folders = [f for f in os.listdir(os.path.join('..', '..', 'data', 'train'))]
for folder in folders:
    if not folder.startswith('.'):
        for file in glob2.glob(os.path.join('..', '..', 'data', 'train', folder, '*.jpg')):
            data = data.append({'FishType':str(folder), 'ImgMatrix':preprocessing_image(file)}, ignore_index=True)

with open('../../data/flattended_img_dump.txt', 'wb') as file:
=======
# Data expected to be in : ../../data/processed/train/
# Script expected to be in ./scripts/data ??
data = pd.DataFrame(columns=('FishType', 'ImgMatrix'))
folders = [f for f in os.listdir(os.path.join('..', '..', 'data', 'processed', 'train'))]
for folder in folders:
    if not folder.startswith('.'):
        for file in glob.glob(os.path.join('..', '..', 'data', 'processed', 'train', folder, '*.jpg')):
            data = data.append({'FishType':str(folder), 'ImgMatrix':preprocessing_image(file)}, ignore_index=True)

with open('../../data/processed/flattended_img_dump.txt', 'wb') as file:
>>>>>>> origin/cut-fish
    pickle.dump(data, file)
