import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Lenna.jpg")
img_b, img_g, img_r = cv2.split(img)

canvs_g = np.zeros((256), np.int)
image_g = np.reshape(img,(-1))

for num in image_g:
    canvs_g[num] += 1

cdf_g = np.zeros((256), np.int)
i = 0
cdf_g[0] = canvs_g[0]
for i in range(1,256): 
    cdf_g[i] = cdf_g[i-1] + canvs_g[i]

cdf1_g = list(cdf_g)
while 0 in cdf1_g:
    cdf1_g.remove(0)


cdf_min_g = min(cdf_g)
cdf_max_g = max(cdf_g)
canvs2_g = np.zeros((316,316),np.uint8)
for row in range(img_g.shape[0]):
    for col in range(img_g.shape[1]):
        # for img[row,col] all pixel(534*800)'s grayLevel value -> equalization
        canvs2_g[row,col] = round((cdf_g[img_g[row,col]]-cdf_min_g) / (cdf_max_g-cdf_min_g) * (256-1))

cv2.imshow('canvs2_g',canvs2_g)
cv2.imwrite('Lenna_g.jpg',canvs2_g)
cv2.waitKey(0)

img2 = cv2.imread("Lenna_g.jpg")
img2_b, img2_g, img2_r = cv2.split(img2)

canvs3_g = np.zeros((256),np.int)
image2_g = np.reshape(img2,(-1))

for num in image2_g:
    canvs3_g[num] += 1

x = range(256)
plt.xlabel('gray Level')
plt.ylabel("number of poxels")
# plt.plot(x, canvs3_g)
plt.bar(x, canvs3_g, align='center', color='g')
plt.show()

