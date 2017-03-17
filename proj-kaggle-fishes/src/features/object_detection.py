tuna = '/Users/mkoutero/Documents/Github/datalab/proj-kaggle-fishes/data/interim/train/crop/train/ALB/img_00201.jpg'
dol = '/Users/mkoutero/Documents/Github/datalab/proj-kaggle-fishes/data/interim/train/crop/train/DOL/img_01185.jpg'
lag = '/Users/mkoutero/Documents/Github/datalab/proj-kaggle-fishes/data/interim/train/crop/train/LAG/img_07774.jpg'
shark = '/Users/mkoutero/Documents/Github/datalab/proj-kaggle-fishes/data/interim/train/crop/train/SHARK/img_02718.jpg'
yft = '/Users/mkoutero/Documents/Github/datalab/proj-kaggle-fishes/data/interim/train/crop/train/YFT/img_01700.jpg'
other = ''

pics = [tuna, dol, lag, shark, yft]

from matplotlib import pyplot as plt
from skimage import exposure
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
from skimage import io
import os

blobs_list = []

for img in pics:
    im = io.imread(img)
    im = exposure.equalize_hist(im)
    image_gray = rgb2gray(im)
    blobs_log = blob_log(image_gray, min_sigma=2, max_sigma=16, num_sigma=5, threshold=.1, overlap=0.1)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    blobs_list.append([im, img, blobs_log])

fig,axes = plt.subplots(2, 3, sharex=False, sharey=False, subplot_kw={'adjustable':'box-forced'})
axes = axes.ravel()
for img, path, blobs in blobs_list:
    ax = axes[0]
    axes = axes[1:]
    ax.set_title(os.path.basename(path))
    ax.imshow(img, interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)

plt.show()

import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.color import rgb2grey, rgb2hsv

levels = []
for img in pics:
    image = io.imread(img)
    thresh = threshold_otsu(image)
    binary = image > thresh
    levels.append([image, thresh, binary])

fig, ax = plt.subplots(5, 3, sharex=False, sharey=False)
index = 0

for image, thresh, binary in levels:
    print index
    # ax = axes[0]
    # axes = axes[1:]
    # ax[index] = plt.subplot(1, 3, 1, adjustable='box-forced')
    # ax[index + 1] = plt.subplot(1, 3, 2)
    # ax[index + 2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')

    ax[index, 0].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 0].set_title('Original')
    ax[index, 0].axis('off')
    ax[index, 0].imshow(image, cmap=plt.cm.gray)

    ax[index, 1].set_title('Histogram')
    ax[index, 1].axvline(thresh, color='r')
    ax[index, 1].hist(image.ravel(), bins=256)

    ax[index, 2].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 2].set_title('Thresholded')
    ax[index, 2].axis('off')
    ax[index, 2].imshow(binary, cmap=plt.cm.gray)
    index += 1

plt.show()



hues = []
for img in pics:
    im = io.imread(img)
    hue = rgb2hsv(im)[:,:,0]
    saturation = rgb2hsv(im)[:,:,1]
    value = rgb2hsv(im)[:,:,2]
    hues.append([os.path.basename(img), hue, saturation, value, im])

fig, ax = plt.subplots(5, 4, sharex=False, sharey=False)
index = 0

for path, hue, saturation, value, im in hues:
    ax[index, 0].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 0].set_title('Original')
    ax[index, 0].axis('off')
    ax[index, 0].imshow(im)

    ax[index, 1].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 1].set_title('Hue')
    ax[index, 1].axis('off')
    ax[index, 1].imshow(hue, cmap=plt.cm.gray)

    ax[index, 2].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 2].set_title('Saturation')
    ax[index, 2].axis('off')
    ax[index, 2].imshow(saturation, cmap=plt.cm.gray)

    ax[index, 3].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 3].set_title('Value')
    ax[index, 3].axis('off')
    ax[index, 3].imshow(value, cmap=plt.cm.gray)

    index += 1


from skimage import feature, measure

cannys = []
for img in pics:
    image = io.imread(img)
    edges = feature.canny(rgb2gray(image), sigma=2, low_threshold=0.9, use_quantiles=True)
    contours = measure.find_contours(edges, 0.8)
    cannys.append([image, edges, contours])


fig, ax = plt.subplots(5, 2, sharex=False, sharey=False)

for index, [image, edges, contours] in enumerate(cannys):
    ax[index, 0].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 0].set_title('Original')
    ax[index, 0].axis('off')
    ax[index, 0].imshow(image, cmap=plt.cm.gray)

    ax[index, 1].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 1].set_title('Canny')
    ax[index, 1].axis('off')
    ax[index, 1].imshow(edges, cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax[index, 1].plot(contour[:, 1], contour[:, 0], linewidth=2)

plt.show()


from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage import measure
import numpy as np
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import blob_log
from skimage.color import rgb2gray
from math import sqrt
import cv2


seg = []
for im in pics:
    img = cv2.imread(im)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    Z = blur.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 16
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((blur.shape))
    #
    b,g,r = cv2.split(res2)
    rgb_img = cv2.merge([r, g, b])
    orb2 = cv2.ORB_create(nfeatures=3000)
    kpF, descsF = orb2.detectAndCompute(img, None)
    blobs_imgFull = cv2.drawKeypoints(img, kpF, None, color=(0,255,0), flags=0)

    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((2,2), np.uint8)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(thresh.shape, np.uint8)
    mask2 = np.zeros(thresh.shape, np.bool)
    for c in contours:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 7000:
            continue
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
    kernel = np.ones((2,2),np.uint8)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask2[mask < 250] = True
    masked = thresh * mask2

    masked_color = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)

    orb = cv2.ORB_create(nfeatures=3000)
    kp, descs = orb.detectAndCompute(res2 * masked_color, None)
    blobs_img = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    # mask = np.zeros(img.copy().shape, np.uint8)
    # for blob in blobs_log:
    #     y, x, r = blob
    #     cv2.circle(mask, (int(x), int(y)), int(r), (255, 255, 255), -1)
    #
    # test = img * mask

    seg.append([img, res2, thresh, masked, blobs_img, blobs_imgFull])

fig, ax = plt.subplots(len(pics), 6, sharex=False, sharey=False)


for index, [img, res2, thresh, masked, blobs_img, blobs_imgFull] in enumerate(seg):
    ax[index, 0].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 0].set_title('Original')
    ax[index, 0].axis('off')
    ax[index, 0].imshow(img)

    ax[index, 1].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 1].set_title('Segmented')
    ax[index, 1].axis('off')
    ax[index, 1].imshow(res2)

    ax[index, 2].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 2].set_title('Threshold')
    ax[index, 2].axis('off')
    ax[index, 2].imshow(thresh, cmap=plt.cm.gray)

    ax[index, 3].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 3].set_title('Threshold+Mask')
    ax[index, 3].axis('off')
    ax[index, 3].imshow(masked, cmap=plt.cm.gray)

    ax[index, 4].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 4].set_title('ORB')
    ax[index, 4].axis('off')
    ax[index, 4].imshow(blobs_img)
    # for blob in blobs:
    #     y, x, r = blob
    #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    #     ax[index, 4].add_patch(c)

    ax[index, 5].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 5].set_title('Original ORB')
    ax[index, 5].axis('off')
    ax[index, 5].imshow(blobs_imgFull)
    # for blob in blobs_imgFull:
    #     y, x, r = blob
    #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    #     ax[index, 5].add_patch(c)

plt.show()

    # ax[index, 3].set_aspect(aspect='auto', adjustable='box-forced')
    # ax[index, 3].set_title('Blob')
    # ax[index, 3].axis('off')
    # ax[index, 3].imshow(img)
    # for blob in blobs:
    #     y, x, r = blob
    #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    #     ax[index, 3].add_patch(c)

    # loop over the contours individually
    # ori = img.copy()
    # for c in contours:
    #     # if the contour is not sufficiently large, ignore it
    #     if cv2.contourArea(c) < 5000:
    #         continue
    #     [x,y,w,h] = cv2.boundingRect(c)
    #     cv2.rectangle(ori,(x,y),(x+w,y+h),(255,0,255),2)

    # ax[index, 4].set_aspect(aspect='auto', adjustable='box-forced')
    # ax[index, 4].set_title('horiz')
    # ax[index, 4].axis('off')
    # ax[index, 4].imshow(edged)





import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread(shark)
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((2,2),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 4)

plt.subplot(131),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(thresh, 'gray')
plt.title("Otus's binary threshold"), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(closing, 'gray')
plt.title("morphologyEx"), plt.xticks([]), plt.yticks([])
plt.show()




import numpy as np
import cv2

img = cv2.imread(yft)
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

plt.imshow(res2)
plt.show()
