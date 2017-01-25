from glob import glob
import os
import shutil
import numpy as np

RAW_DIR = '../../data/raw/'
INT_DIR = '../../data/interim/'
DEVCROP_DIR = INT_DIR + 'train/devcrop/'

# Creation DevCrop
if not os.path.isdir(DEVCROP_DIR):
    os.makedirs(DEVCROP_DIR)

# Create train
if not os.path.isdir(DEVCROP_DIR + 'train'):
    shutil.copytree(RAW_DIR + 'train', DEVCROP_DIR + 'train')
else: 
    print('Rewriting existing ' + DEVCROP_DIR + 'train')
    shutil.rmtree(DEVCROP_DIR + 'train')
    shutil.copytree(RAW_DIR + 'train', DEVCROP_DIR + 'train')

# Create valid
if not os.path.isdir(DEVCROP_DIR + 'valid'):
    os.mkdir(DEVCROP_DIR + 'valid')
else: 
    print("Rewriting existing " + DEVCROP_DIR + 'valid')
    shutil.rmtree(DEVCROP_DIR + 'valid')
    os.mkdir(DEVCROP_DIR + 'valid')

# Create test
if not os.path.isdir(DEVCROP_DIR + 'test'):
    os.mkdir(DEVCROP_DIR + 'test')
    shutil.copytree(RAW_DIR + 'test', DEVCROP_DIR + 'test/category')
else: print("Already existing " + DEVCROP_DIR + 'test')

os.chdir(DEVCROP_DIR + 'train')
g = glob('*')
for d in g: 
    if not os.path.isdir('../valid/'+d):
        os.mkdir('../valid/'+d)
        print('../valid/'+d)
    else: print("Existing valid/"+d)

g = glob('*/*.jpg')
print("Number of fish in train = ",len(g))
shuf = np.random.permutation(g)
for i in range(500):
    os.rename(shuf[i], '../valid/' + shuf[i])

