import pandas as pd
from PIL import Image
import PIL
import os
from glob import glob
import cv2
from matplotlib import pyplot as plt


#from utils import *

MODELS = "../../models/"
RAW_DIR = '../../data/raw/'
PATH = "../../data/interim/"


bb_box = pd.read_csv(MODELS + 'bbox.csv')
#test = get_data(PATH+'test')
#test_batches = get_batches(PATH+'test', shuffle=False, batch_size=1)


def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2]-10, bb[3]-10), bb[1]+30, bb[0]+30, color=color, fill=False, lw=3)


os.chdir(RAW_DIR + 'test')
g = glob('*')

sizes = [PIL.Image.open(f).size for f in g]

#bb_box.loc[bb_box['image'] == 'img_00943.jpg']
#bb_box.loc[bb_box['image'] == g[12]]
#a = bb_box.loc[bb_box['image'] == g[13]].iloc[:,[1,2,3,4]].values


# Convert dictionary into array
bb_params = ['height', 'width', 'x', 'y']
def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (224. / size[0])
    conv_y = (224. / size[1])
    bb[0] = bb[0]/conv_y
    bb[1] = bb[1]/conv_x
    bb[2] = max(bb[2]/conv_x, 0)
    bb[3] = max(bb[3]/conv_y, 0)
    return bb

trn_bbox = np.stack([convert_bb(bb_json[f], s) for f,s in zip(raw_filenames, sizes)],).astype(np.float32)

i=16
bb = bb_box.loc[bb_box['image'] == g[i]].iloc[:,[1,2,3,4]].values
#bb = [100,100,100,100]
im = cv2.imread(g[i])
im = cv2.resize(im, (224, 224))
plt.imshow(im)
ax=plt.gca()
ax.add_patch(create_rect(bb[0,]))
name="FishS%i.png" % i
plt.savefig('../../../src/visualization/'+name)
plt.cla()


print type(im)

im = Image.open(g)
im.save(file + ".thumbnail", "JPEG")

g[1]
plot(test[1])
plt.imshow(np.rollaxis(test[1], 0, 3).astype(np.uint8))
plot(val[i])
ax=plt.gca()
ax.add_patch(create_rect(bb_pred, 'yellow'))
name="FishS%i.png" % 2
plt.savefig(name)
plt.cla()


size = 128, 128

for infile in glob.glob("*.jpg"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im.thumbnail(size, Image.ANTIALIAS)
    im.save(file + ".thumbnail", "JPEG")


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




def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def show_bb_pred(i):
    bb = val_bbox[i]
    bb_pred = pred[0][i]
    plt.figure(figsize=(6,6))
    plot(val[i])
    ax=plt.gca()
    ax.add_patch(create_rect(bb_pred, 'yellow'))
    ax.add_patch(create_rect(bb))
    name="FishS%i.png" % i
    plt.savefig(name)
    plt.cla()


return np.rollaxis(img, 0, 3).astype(np.uint8)
def plot(img):
    plt.imshow(to_plot(img))

