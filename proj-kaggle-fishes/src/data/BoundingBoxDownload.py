# Bounding box labels of the fishes has pubished in the forums
# https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/25902/complete-bounding-box-annotation

import os
import urllib2

LABELS_DIR = '../../data/external/labels/'

# Links to labels produced by Nathaniel Shimoni, thanks for the great work!
ANNOS_DIR = '../../data/external/annos/'

# Label of shark and yft have a different labeling

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
    download_labels(ANNOS_DIR, ANNOS_LINKS)
