import glob
import os
import json
import cv2
import numpy as np
import shutil
import pickle

#from ..data import DataModel

ANNOS_DIR = '../../data/external/annos/'
OUTPUT_DIR = '../../data/interim/train/crop/train/'
TRAIN_DIR = '../../data/raw/train/'

RELABELS_PATH = "../../data/external/relabels/relabels.csv"

def process_annos(label_file):
    file_name = os.path.basename(label_file)
    class_name = file_name.split("_")[0]
    if not os.path.isdir(OUTPUT_DIR + class_name.upper()):
        os.mkdir(OUTPUT_DIR + class_name.upper())
    print("Processing " + class_name + " labels")
    with open(label_file) as data_file:
        data = json.load(data_file)

    for img_data in data:
        # Shark and yft have different 'filename'
        if class_name == "shark" or class_name == "yft" or class_name == "OTHER": 
            img_file = TRAIN_DIR + class_name.upper() + '/' + img_data['filename'].split("/")[-1]
        else: img_file = TRAIN_DIR + class_name.upper() + '/' + img_data['filename']
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

            #print(img_file.split("/")[-1], x_left,x_right,y_up,y_down,diff )
            img = img[y_up:y_down, x_left:x_right, :]
            #img = img[y0:y1, x0:x1, :]
#            p_tails = (img_data['annotations'][0]['x'], img_data['annotations'][0]['y'])
#            p_heads = (img_data['annotations'][0]['x'] +
#                    img_data['annotations'][0]['width'],
#                    img_data['annotations'][0]['y'] +
#                    img_data['annotations'][0]['height']) 
#            p_middle = ((p_heads[0] + p_tails[0]) / 2, (p_heads[1] +
#                p_tails[1]) / 2)
#            dist = np.sqrt((p_heads[0] - p_tails[0]) ** 2 + (p_heads[1] - p_tails[1]) ** 2)
#            offset = 3.0 * dist / 4.0
#            img_width = img.shape[1]
#            img_height = img.shape[0]
#            x_left = max(0, p_middle[0] - offset)
#            x_right = min(img_width - 1, p_middle[0] + offset)
#            y_up = max(0, p_middle[1] - offset)
#            y_down = min(img_height - 1, p_middle[1] + offset)
#            x_left, x_right, y_up, y_down = int(x_left), int(x_right), int(y_up), int(y_down)
#            img = img[y_up:y_down+1, x_left:x_right+1, :]
            if class_name == "shark" or class_name == "yft" or class_name == "OTHER": 
                cv2.imwrite(OUTPUT_DIR + class_name.upper() + '/' +
                        img_data['filename'].split("/")[-1], img)
            else: cv2.imwrite(OUTPUT_DIR + class_name.upper() + '/' +
                    img_data['filename'], img)
        else:
            if class_name == "shark" or class_name == "yft" or class_name == "OTHER":
                shutil.copyfile(TRAIN_DIR + class_name.upper() + '/' +
                        img_data['filename'].split("/")[-1], OUTPUT_DIR + class_name.upper() +
                        '/' + img_data['filename'].split("/")[-1])
            else:
                shutil.copyfile(TRAIN_DIR + class_name.upper() + '/' +
                        img_data['filename'], OUTPUT_DIR + class_name.upper() +
                        '/' + img_data['filename'])

def make_cropped_dataset():
    label_files = glob.glob(ANNOS_DIR + '*.json')
    for file in label_files:
        process_annos(file)

def copy_nofish():
    if not os.path.isdir(OUTPUT_DIR + "NoF"):
        shutil.copytree(TRAIN_DIR + "NoF" , OUTPUT_DIR + 'NoF')
    #if not os.path.isdir(OUTPUT_DIR + "OTHER"):
    #    shutil.copytree(TRAIN_DIR + "OTHER" , OUTPUT_DIR + 'OTHER')


def separate_val():
    print("separate val")
    df = pickle.load(open('../../data/processed/df.txt', 'rb'))
    for img,boat in zip(df["img_file"],df["boat_group"]):
        #print(j.split("/")[-1])
        classes = img.split("/")[-2]
        basename = img.split("/")[-1]
        if boat == 1 or boat == 5 or boat == 7 or boat == 8 or boat == 11:
            print(basename,"  ",boat)
            fromname='../../data/interim/train/crop/train/' + classes 
            toname='../../data/interim/train/crop/val/'+ classes 
            if not os.path.isdir(toname):
                os.makedirs(toname)

            try: 
                os.rename(fromname + '/' + basename,toname + '/' + basename)
            except: 
                pass

def relabel():
    print("relabel")
    #os.mkdir("{}/{}".format(OUTPUT_DIR, "revise"))
    with open(RELABELS_PATH) as f:
        for line in f:
            cols = line.split()
            src = "{}{}/{}.jpg".format(OUTPUT_DIR, cols[1], cols[0])
            dst = "{}{}/{}.jpg".format(OUTPUT_DIR, cols[2], cols[0])

            try:
                os.rename(src, dst)
                print("{} ".format(src))

            except FileNotFoundError:
                print("{} not found".format(src))


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    make_cropped_dataset()
    copy_nofish()
    relabel()
    separate_val()
