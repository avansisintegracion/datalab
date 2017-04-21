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

# Create test
if not os.path.isdir(DEVCROP_DIR + 'test'):
    os.mkdir(DEVCROP_DIR + 'test')
    shutil.copytree(RAW_DIR + 'test', DEVCROP_DIR + 'test/category')
else: print("Already existing " + DEVCROP_DIR + 'test')

# Create sample folder
if not os.path.isdir(DEVCROP_DIR + 'sample'):
    os.mkdir(DEVCROP_DIR + 'sample')
    os.mkdir(DEVCROP_DIR + 'sample/train')
    os.mkdir(DEVCROP_DIR + 'sample/test')
else:
    print("Already existing " + DEVCROP_DIR + 'sample')
    shutil.rmtree(DEVCROP_DIR + 'sample')
    os.mkdir(DEVCROP_DIR + 'sample')
    os.mkdir(DEVCROP_DIR + 'sample/train')
    os.mkdir(DEVCROP_DIR + 'sample/test')

# Create classes folders
os.chdir(DEVCROP_DIR + 'train')
g = glob('*')
for d in g:
    if not os.path.isdir('../val/'+d):
        os.makedirs('../val/'+d)
        print('../val/'+d)
    else: print("Existing valid/"+d)

g = glob('*/*.jpg')
print("Number of fish in train = ",len(g))
shuf = np.random.permutation(g)
for i in range(500):
    print(shuf[i], '../val/' + shuf[i])
    shutil.copyfile(shuf[i], '../val/' + shuf[i])

# Create val folders
#os.chdir(DEVCROP_DIR + 'train')
g = glob('*')
for d in g:
    if not os.path.isdir('../sample/train/'+d):
        os.mkdir('../sample/train/'+d)
        print('../sample/train/'+d)
    else: print("Existing valid/"+d)

g = glob('*/*.jpg')
print("Number of fish in train = ",len(g))
shuf = np.random.permutation(g)
for i in range(500):
    print(shuf[i], '../sample/train/' + shuf[i])
    shutil.move(shuf[i], '../sample/train/' + shuf[i])

# Create classes folders
os.chdir('../test/category')
if not os.path.isdir('../../sample/test/category'):
    os.mkdir('../../sample/test/category')
    print('../../sample/test/category')
else: print("Existing test/category")

g = glob('*.jpg')
print("Number of fish in train = ",len(g))
shuf = np.random.permutation(g)
for i in range(200):
    print(shuf[i], '../../sample/test/category/' + shuf[i])
    shutil.copyfile(shuf[i], '../../sample/test/category/' + shuf[i])
