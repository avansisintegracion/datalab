import glob
import os
import json
import cv2
import numpy as np
import shutil
import pickle

ANNOS_DIR = '../../data/external/annos/'
OUTPUT_DIR = '../../data/interim/train/crop/train/'
TRAIN_DIR = '../../data/raw/train/'


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
        if class_name == "shark" or class_name == "yft": 
            img_file = TRAIN_DIR + class_name.upper() + '/' + img_data['filename'].split("/")[-1]
        else: img_file = TRAIN_DIR + class_name.upper() + '/' + img_data['filename']
        img = cv2.imread(img_file)
        # Crop only images with both heads and tails present for cleaner dataset
        if len(img_data['annotations']) >= 1:
            p_tails = (img_data['annotations'][0]['x'], img_data['annotations'][0]['y'])
            p_heads = (img_data['annotations'][0]['x'] +
                    img_data['annotations'][0]['width'],
                    img_data['annotations'][0]['y'] +
                    img_data['annotations'][0]['height']) 
            p_middle = ((p_heads[0] + p_tails[0]) / 2, (p_heads[1] +
                p_tails[1]) / 2)
            dist = np.sqrt((p_heads[0] - p_tails[0]) ** 2 + (p_heads[1] - p_tails[1]) ** 2)
            offset = 3.0 * dist / 4.0
            img_width = img.shape[1]
            img_height = img.shape[0]
            x_left = max(0, p_middle[0] - offset)
            x_right = min(img_width - 1, p_middle[0] + offset)
            y_up = max(0, p_middle[1] - offset)
            y_down = min(img_height - 1, p_middle[1] + offset)
            x_left, x_right, y_up, y_down = int(x_left), int(x_right), int(y_up), int(y_down)
            img = img[y_up:y_down+1, x_left:x_right+1, :]
            if class_name == "shark" or class_name == "yft": 
                cv2.imwrite(OUTPUT_DIR + class_name.upper() + '/' +
                        img_data['filename'].split("/")[-1], img)
            else: cv2.imwrite(OUTPUT_DIR + class_name.upper() + '/' +
                    img_data['filename'], img)
        else:
            if class_name == "shark" or class_name == "yft":
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

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    make_cropped_dataset()
    copy_nofish()
    separate_val()
