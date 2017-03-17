
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage import io
import os
from skimage import feature, measure, segmentation

PATH='/Users/mkoutero/Documents/Github/datalab/proj-kaggle-fishes/data/interim/train/crop/train/'
tuna = PATH + 'ALB/img_00201.jpg'
dol = PATH + 'DOL/img_01185.jpg'
lag = PATH + 'LAG/img_07774.jpg'
shark = PATH + 'SHARK/img_02718.jpg'
yft = PATH + 'YFT/img_01700.jpg'
other = ''

pics = [tuna, dol, lag, shark, yft]


cannys = []
for img in pics:
    image = io.imread(img)
    edges = feature.canny(rgb2gray(image), sigma=2, low_threshold=0.9, use_quantiles=True)
    contours = measure.find_contours(edges, 0.8)
    seg = segmentation.find_boundaries(edges)
    cannys.append([image, edges, contours, seg])


fig, ax = plt.subplots(5, 3, sharex=False, sharey=False)
index = 0

for index, [image, edges, contours, seg] in enumerate(cannys):
    ax[index, 0].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 0].set_title('Original')
    ax[index, 0].axis('off')
    ax[index, 0].imshow(image)

    ax[index, 1].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 1].set_title('Canny')
    ax[index, 1].axis('off')
    ax[index, 1].imshow(edges, cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax[index, 1].plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax[index, 2].set_aspect(aspect='auto', adjustable='box-forced')
    ax[index, 2].set_title('Find boundaries')
    ax[index, 2].axis('off')
    ax[index, 2].imshow(seg, cmap=plt.cm.gray)

plt.show()
