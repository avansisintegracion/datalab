# One has pubished in the forums the bounding box of the fishes
# https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/25902/complete-bounding-box-annotation

import os
import urllib2

LABELS_DIR = '../../data/external/labels/'
# Labels are not complete, annos instead have the figure for all the fish in train set
ANNOS_DIR = '../../data/external/annos/'

# Links to labels produced by Nathaniel Shimoni, thanks for the great work!
LABELS_LINKS = [
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5373/yft_labels.json',
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5374/shark_labels.json',
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5375/lag_labels.json',
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5376/dol_labels.json',
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5377/bet_labels.json',
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5378/alb_labels.json',
]

ANNOS_LINKS = [
        'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5458/bet_labels.json',
        'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5459/shark_labels.json',
        'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5460/dol_labels.json',
        'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5461/yft_labels.json',
        'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5462/alb_labels.json',
        'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5463/lag_labels.json',
]

def download_labels(fichier, httplink):
    if not os.path.isdir(fichier):
        os.mkdir(fichier)
    for link in httplink:
        label_filename = link.split('/')[-1]
        print("Downloading " + label_filename)
        f = urllib2.urlopen(link)
        with open(fichier + label_filename, 'wb') as local_file:
            local_file.write(f.read())

if __name__ == '__main__':
    download_labels(LABELS_DIR, LABELS_LINKS)
    download_labels(ANNOS_DIR, ANNOS_LINKS)
