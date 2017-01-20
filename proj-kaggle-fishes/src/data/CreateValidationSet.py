from glob import glob
import os
import shutil
import numpy as np

RAW_DIR = '../../data/raw/'
INT_DIR = '../../data/interim/'

# Creation 
if not os.path.isdir(INT_DIR + 'train'):
    shutil.copytree(RAW_DIR + 'train', INT_DIR + 'train')
else: 
    print('Already existing ' + INT_DIR + 'train')
    shutil.rmtree(INT_DIR + 'train')
    shutil.copytree(RAW_DIR + 'train', INT_DIR + 'train')

if not os.path.isdir(INT_DIR + 'valid'):
    os.mkdir(INT_DIR + 'valid')
else: 
    print("Already existing " + INT_DIR + 'valid')
    shutil.rmtree(INT_DIR + 'valid')
    os.mkdir(INT_DIR + 'valid')

if not os.path.isdir(INT_DIR + 'test'):
    os.mkdir(INT_DIR + 'test')
    shutil.copytree(RAW_DIR + 'test', INT_DIR + 'test/category')
else: print("Already existing " + INT_DIR + 'test')

os.chdir(INT_DIR + 'train')
g = glob('*')
for d in g: 
    if not os.path.isdir('../../interim/valid/'+d):
        os.mkdir('../../interim/valid/'+d)
        print('../../interim/valid/'+d)
    else: print("Existing valid/"+d)

g = glob('*/*.jpg')
print("Number of fish in train = ",len(g))
shuf = np.random.permutation(g)
for i in range(500):
    os.rename(shuf[i], '../valid/' + shuf[i])

