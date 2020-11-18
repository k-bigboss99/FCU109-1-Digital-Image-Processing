import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("cat.jpg",cv2.IMREAD_GRAYSCALE)

canvs = np.zeros((256), np.int)
image = np.reshape(img,(-1))

for num in image:
    canvs[num] += 1

cdf = np.zeros((256), np.int)

i = 0
cdf[0] = canvs[0]
for i in range(1,256): # 1-255
    cdf[i] = cdf[i-1] + canvs[i]

# 去除 灰階值 數量為 0 
# ndArray != list
cdf1 = list(cdf)
while 0 in cdf1:
    cdf1.remove(0)

cdf_min = min(cdf)
cdf_max = max(cdf)
canvs2 = np.zeros((534,800),np.uint8)
for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        # for img[row,col] all pixel(534*800)'s grayLevel value -> equalization
        canvs2[row,col] = round((cdf[img[row,col]]-cdf_min) / (cdf_max-cdf_min) * (256-1))

cv2.imshow('canvs2', canvs2)
cv2.imwrite('cat2.jpg', canvs2)
cv2.waitKey(0)

img2 = cv2.imread("cat2.jpg", cv2.IMREAD_GRAYSCALE)

# cv2.imshow("gray",img2)

cv2.waitKey()

canvs3 = np.zeros((256), np.int)
image2 = np.reshape(img2, (-1))
# 計算向素值的數量
for num in image2:
    canvs3[num] += 1


x = range(256)
plt.xlabel('gray Level')
plt.ylabel("number of poxels")
# plt.plot(x, canvs3)
plt.bar(x, canvs3, align='center', color='b')
plt.show()


