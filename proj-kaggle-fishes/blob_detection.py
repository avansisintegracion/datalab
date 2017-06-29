# Standard imports
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Read image
im = cv2.imread("/Users/mkoutero/Documents/Github/datalab/proj-kaggle-fishes/notebooks/3537255216_d766eac288.jpg")

blur = cv2.GaussianBlur(im, (3, 3), 0)
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
# Convert to grayscale and apply otsu.
gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

# Noise removal by contour detection of large elements
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
mask = np.zeros(thresh.shape, np.uint8)
mask2 = np.zeros(thresh.shape, np.bool)
# Remove large elements, typically boat structures
for c in contours:
    # if the contour is not sufficiently large, ignore it
    # this parameter is highly dependant on the image size
    if cv2.contourArea(c) < 20000:
        continue
    cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
mask2[mask < 250] = True
masked = thresh * mask2
masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)

# Change thresholds
params.minThreshold = 2;
params.maxThreshold = 2000;

# Filter by Area.
params.filterByArea = True
params.minArea = 10

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.001

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.05

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.000001

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else :
    detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(res2 * masked)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# The important part - Correct BGR to RGB channel
#im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#im_with_keypoints = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB)

# Plot shape of the mask and the detected keypoints
fig, ax = plt.subplots(1, 2, sharex=False, sharey=False)
ax[0].set_aspect(aspect='auto', adjustable='box-forced')
ax[0].set_title('Before')
ax[0].axis('off')
ax[0].imshow(im, cmap=plt.cm.gray)

ax[1].set_aspect(aspect='auto', adjustable='box-forced')
ax[1].set_title('After')
ax[1].axis('off')
ax[1].imshow(im_with_keypoints)
plt.show()
