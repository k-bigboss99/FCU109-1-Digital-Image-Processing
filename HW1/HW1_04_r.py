import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Lenna.jpg")
img_b, img_g, img_r = cv2.split(img)

canvs_r = np.zeros((256), np.int)
image_r = np.reshape(img,(-1))

for num in image_r:
    canvs_r[num] += 1

cdf_r = np.zeros((256), np.int)

i = 0
cdf_r[0] = canvs_r[0]
for i in range(1,256): 
    cdf_r[i] = cdf_r[i-1] + canvs_r[i]

cdf1_r = list(cdf_r)
while 0 in cdf1_r:
    cdf1_r.remove(0)


cdf_min_r = min(cdf_r)
cdf_max_r = max(cdf_r)
canvs2_r = np.zeros((316,316),np.uint8)
for row in range(img_r.shape[0]):
    for col in range(img_r.shape[1]):
        # for img[row,col] all pixel(534*800)'s grayLevel value -> equalization
        canvs2_r[row,col] = np.round((cdf_r[img_r[row,col]]-cdf_min_r) / (cdf_max_r-cdf_min_r) * (256-1))

cv2.imshow('canvs2_r',canvs2_r)
cv2.imwrite('Lenna_r.jpg',canvs2_r)
cv2.waitKey(0)

img2 = cv2.imread("Lenna_r.jpg")
img2_b, img2_g, img2_r = cv2.split(img2)

canvs3_r = np.zeros((256),np.int)
image2_r = np.reshape(img2,(-1))

for num in image2_r:
    canvs3_r[num] += 1
print(canvs3_r)


# 直方圖
x = range(256)
plt.xlabel('gray Level')
plt.ylabel("number of poxels")
# plt.plot(x, canvs3_r)
plt.bar(x, canvs3_r, align='center', color='r')
plt.show()


