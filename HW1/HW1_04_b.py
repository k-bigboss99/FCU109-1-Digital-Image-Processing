import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Lenna.jpg")
img_b, img_g, img_r = cv2.split(img)

canvs_b = np.zeros((256), np.int)
image_b = np.reshape(img,(-1))

for num in image_b:
    canvs_b[num] += 1

tmp_b = np.zeros((256), np.int)

i = 0
tmp_b[0] = canvs_b[0]
for i in range(1,256): 
    tmp_b[i] = tmp_b[i-1] + canvs_b[i]

tmp1_b = list(tmp_b)
while 0 in tmp1_b:
    tmp1_b.remove(0)

cdf_b = tmp_b / tmp_b[255]
cdf_min_b = min(cdf_b)
cdf_max_b = max(cdf_b)
canvs2_b = np.zeros((316,316),np.uint8)
for row in range(img_b.shape[0]):
    for col in range(img_b.shape[1]):
        # for img[row,col] all pixel(534*800)'s grayLevel value -> equalization
        canvs2_b[row,col] = np.round((cdf_b[img_b[row,col]]-cdf_min_b) / (cdf_max_b-cdf_min_b) * (256-1))

cv2.imshow('canvs2_b',canvs2_b)
cv2.imwrite('Lenna_b.jpg',canvs2_b)
cv2.waitKey(0)

img2 = cv2.imread("Lenna_b.jpg")
img2_b,img2_g,img2_r=cv2.split(img2)


canvs3_b = np.zeros((256),np.int)
image2_b = np.reshape(img2,(-1))

for num in image2_b:
    canvs3_b[num] += 1

    

x = range(256)
plt.xlabel('gray Level')
plt.ylabel("number of poxels")
# plt.plot(x, canvs3_b)
plt.bar(x, canvs3_b, align='center', color='b')
plt.show()


