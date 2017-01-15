# One has pubished in the forums the bounding box of the fishes
# https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/25902/complete-bounding-box-annotation

import os
import urllib2

LABELS_DIR = 'labels/'

# Links to labels produced by Nathaniel Shimoni, thanks for the great work!
LABELS_LINKS = [
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5373/yft_labels.json',
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5374/shark_labels.json',
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5375/lag_labels.json',
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5376/dol_labels.json',
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5377/bet_labels.json',
    'https://www.kaggle.com/blobs/download/forum-message-attachment-files/5378/alb_labels.json',
]

def download_labels():
    if not os.path.isdir(LABELS_DIR):
        os.mkdir(LABELS_DIR)
    for link in LABELS_LINKS:
        label_filename = link.split('/')[-1]
        print("Downloading " + label_filename)
        f = urllib2.urlopen(link)
        with open(LABELS_DIR + label_filename, 'wb') as local_file:
            local_file.write(f.read())

if __name__ == '__main__':
    download_labels()
