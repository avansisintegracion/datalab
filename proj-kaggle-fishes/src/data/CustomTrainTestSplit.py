
import numpy as np
import pandas as pd
import glob
import cv2
import time
import os
import pickle
from collections import Counter
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import KMeans


def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (32, 32))
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB',
               'BET',
               'DOL',
               'LAG',
               'SHARK',
               'YFT',
               'OTHER',
               'NoF'
               ]
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', '..', 'data', 'raw', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(fl)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


train_x, train_y, train_id = read_and_normalize_train_data()

np_x = np.array(train_x)

data_final = np_x.reshape(3777, 3072)
data_final

n_components = 50
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(data_final)

x_train_pca = pca.transform(data_final)

n_boats = 14

kmeans = KMeans(n_clusters=n_boats, random_state=0).fit(x_train_pca)

predicted_labels = kmeans.predict(x_train_pca)

predicted_labels


df = pd.DataFrame({'img_file': train_id,
                   'boat_group': predicted_labels,
                   'labels': train_y})


with open('../../data/processed/df.txt', 'wb') as file:
    pickle.dump(df, file)

for cat in range(0, 8):
    tmp = df.loc[df['labels'] == cat]
    print('This is cat : %i' % cat)
    counts = Counter(tmp['boat_group'])
    print(counts)

# Manual selection of categories that allow a 80/20 split
# that keeps certain boats from being in the subtrain dataset.
# CHANGE the boatID everytime the script is run to accomodate
df_20 = df.loc[df['boat_group'].isin([1, 5, 7, 8, 11])]
df_80 = df.loc[~df['boat_group'].isin([1, 5, 7, 8, 11])]


with open('../../data/processed/df_20.txt', 'wb') as file:
    pickle.dump(df_20, file)

with open('../../data/processed/df_80.txt', 'wb') as file:
    pickle.dump(df_80, file)
