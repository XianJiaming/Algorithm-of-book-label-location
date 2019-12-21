import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import math
from busyCal import findLines

img = cv2.imread('books2.jpg', cv2.IMREAD_ANYCOLOR)
img = cv2.resize(img, (1000, 800))
img_b = img[:, :, 0]
img_g = img[:, :, 1]
img_r = img[:, :, 2]
purple = (img_r > img_g) * (img_b > img_g) * ((np.int8(img_r) - np.int8(img_b)) < 15)

filter_img = img.copy()
filter_img[purple] = [200, 100, 0]
cv2.imwrite('filter.jpg',filter_img)
# purple = np.where(cv2.Canny(img, 100, 100) * purple)
dst_g = cv2.cornerHarris(img_g, 7, 13, 0.01)
dst_r = cv2.cornerHarris(img_r, 7, 13, 0.01)
dst_b = cv2.cornerHarris(img_b, 7, 13, 0.01)
dst = (dst_g > dst_g.max() * 0.01) + (dst_r > dst_r.max() * 0.01) + (dst_b > dst_b.max() * 0.01)
filter_img = img.copy()
filter_img[dst] = [200,100,0]
cv2.imwrite('filter2.jpg',filter_img)

purple[dst == False] = False
filter_img = img.copy()
filter_img[purple] = [200,100,0]
cv2.imwrite('filter3.jpg',filter_img)

filter_img[:] = [0,0,0]
filter_img[purple] = [255,255,255]
cv2.imwrite('filter4.jpg',filter_img)

# finePoint = neighIsWhite(img, purple)
purple = np.where(purple)

# cv2.imshow('sd',img);cv2.waitKey()


purple = np.array(list(zip(purple[0], purple[1])))
db = DBSCAN(eps=55, min_samples=200).fit(purple)

# 找到一个据点，合成各个角点
purple = purple[db.labels_ >= 0]
filter_img[:] = [0,0,0]
filter_img[purple[:, 0], purple[:, 1]] = [255,255,255]
cv2.imwrite('filter5.jpg',filter_img)


db = DBSCAN(eps=3, min_samples=3).fit(purple)
centure = []
for i in range(db.labels_.max() + 1):
    clus = purple[db.labels_ == i]
    centure.append(sum(clus) // clus.shape[0])

filter_img = img.copy()
for c1 in centure:
    cv2.circle(filter_img, (c1[1], c1[0]), 10, (255, 100, 55), 2)
cv2.imwrite('filter6.jpg',filter_img)

for c1 in centure:
    cv2.circle(img, (c1[1], c1[0]), 10, (0, 0, 255), 2)

lines = findLines(centure, math.pi / 6)
lines.sort(key=lambda x: len(x.points), reverse=True)

firstLine = [list(p) for p in lines[0].points]

for line in lines[1:]:
    secondLine = [list(p) for p in line.points]
    for p in firstLine:
        if p in secondLine:
            break
    else:
        break

for c1 in firstLine:
    cv2.circle(img, (c1[1], c1[0]), 10, (0, 100, 55), 2)
for c2 in secondLine:
    cv2.circle(img, (c2[1], c2[0]), 10, (100, 0, 55), 2)

# img[purple[:,0],purple[:,1]] = [100,200,55]
cv2.imwrite('ts2.jpg', img)
cv2.imshow('t', img)
cv2.waitKey()

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = purple[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = purple[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.show()
