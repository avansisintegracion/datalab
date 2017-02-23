import cv2
import glob
import numpy as np
import os
import shutil

OTSU_DIR = '../../data/interim/train/cropotsu/'
CROP_DIR = '../../data/interim/train/crop/'

def get_holes(image, thresh):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    im_bw_inv = cv2.bitwise_not(im_bw)

    contour, _ = cv2.findContours(im_bw_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(im_bw_inv, [cnt], 0, 255, -1)

    nt = cv2.bitwise_not(im_bw)
    im_bw_inv = cv2.bitwise_or(im_bw_inv, nt)
    return im_bw_inv


def remove_background(image, thresh, scale_factor=.25, kernel_range=range(1, 15), border=None):
    border = border or kernel_range[-1]

    holes = get_holes(image, thresh)
    small = cv2.resize(holes, None, fx=scale_factor, fy=scale_factor)
    bordered = cv2.copyMakeBorder(small, border, border, border, border, cv2.BORDER_CONSTANT)

    for i in kernel_range:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*i+1, 2*i+1))
        bordered = cv2.morphologyEx(bordered, cv2.MORPH_CLOSE, kernel)

    unbordered = bordered[border: -border, border: -border]
    mask = cv2.resize(unbordered, (image.shape[1], image.shape[0]))
    fg = cv2.bitwise_and(image, image, mask=mask)
    return fg


# Creation CropOtsu
if not os.path.isdir(OTSU_DIR):
    os.makedirs(OTSU_DIR)
    os.makedirs(OTSU_DIR + 'train')
    os.makedirs(OTSU_DIR + 'val')

os.chdir(CROP_DIR + 'train')
g = glob.glob('*')
for d in g: 
    if not os.path.isdir('../../cropotsu/train/'+d):
        os.mkdir('../../cropotsu/train/'+d)
        print('../../cropotsu/train/'+d)
    else: print("Existing ../../cropotsu/"+d)

g = glob.glob('*/*.jpg')
print("Number of fish in train = ",len(g))
for i in g:
    #shutil.copyfile(i, '../../cropotsu/val/' + i)
    raw_name = i.split("/")[-1]
    img = cv2.imread(i)
    #print(img.dtype)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(img_grey,0,255,cv2.THRESH_OTSU)[0] - 30
    #print(otsu)
    nb_img = remove_background(img, otsu)
    name = '../../cropotsu/train/' + i
    #print(name)
    cv2.imwrite(name,nb_img)
    del img, nb_img

os.chdir('../val')
g = glob.glob('*')
for d in g: 
    if not os.path.isdir('../../cropotsu/val/'+d):
        os.mkdir('../../cropotsu/val/'+d)
        print('../../cropotsu/val/'+d)
    else: print("Existing ../../cropotsu/"+d)

g = glob.glob('*/*.jpg')
print("Number of fish in val = ",len(g))
for i in g:
    #shutil.copyfile(i, '../../cropotsu/val/' + i)
    raw_name = i.split("/")[-1]
    img = cv2.imread(i)
    #print(img.dtype)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(img_grey,0,255,cv2.THRESH_OTSU)[0] - 30
    #print(otsu)
    nb_img = remove_background(img, otsu)
    name = '../../cropotsu/val/' + i
    #print(name)
    cv2.imwrite(name,nb_img)
    del img, nb_img

#photos = glob.glob('sample/*.jpg')
#
#for i in photos:
#    raw_name = i.split("/")[-1]
#    img = cv2.imread(i)
#    print(img.dtype)
#    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    otsu = cv2.threshold(img_grey,0,255,cv2.THRESH_OTSU)[0] - 30
#    print(otsu)
#    nb_img = remove_background(img, otsu)
#    name = 'C' + raw_name
#    print(name)
#    cv2.imwrite(name,nb_img)
#    del img, nb_img

#img = cv2.imread('img_00085.jpg')
#nb_img = remove_background(img, 70)
#cv2.imwrite('img_test.jpg',nb_img)
