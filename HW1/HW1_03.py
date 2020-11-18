import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Lenna.jpg")

cv2.imshow("Lenna.jpg", img)
# cv2.Waitkey([delay]) 等待按鍵
cv2.waitKey()
img_b, img_g, img_r=cv2.split(img)

# 顯示 img b 通道
canvs_b2 = np.zeros((316,316,3),np.uint8)
canvs_b2[:,:,0] = img_b
cv2.imwrite('canvs_b2.jpg',canvs_b2)
image = cv2.imread('canvs_b2.jpg')
cv2.imshow("canvs_b2",image)
cv2.waitKey(0)


# 顯示 img g 通道
canvs_g2 = np.zeros((316,316,3),np.uint8)
canvs_g2[:,:,1] = img_g
cv2.imwrite('canvs_g2.jpg',canvs_g2)
image = cv2.imread('canvs_g2.jpg')
cv2.imshow("canvs_g2",image)
cv2.waitKey(0)


# 顯示 img r 通道
canvs_r2 = np.zeros((316,316,3),np.uint8)
canvs_r2[:,:,2] = img_r
cv2.imwrite('canvs_r2.jpg',canvs_r2)
image = cv2.imread('canvs_r2.jpg')
cv2.imshow("canvs_r2",image)
cv2.waitKey(0)

canvs_b = np.zeros((256), np.int)
canvs_g = np.zeros((256), np.int)
canvs_r = np.zeros((256), np.int)

image_b = np.reshape(img_b,(-1))
image_g = np.reshape(img_g,(-1))
image_r = np.reshape(img_r,(-1))

for num in image_b:
    canvs_b[num] += 1

for num in image_g:
    canvs_g[num] += 1

for num in image_r:
    canvs_r[num] += 1

x = range(256)
plt.xlabel('gray Level')
plt.ylabel("number of poxels")
# plt.plot(x, canvs_b,'b')
plt.bar(x, canvs_b, align='center', color='b')
plt.show()
# plt.plot(x, canvs_g,'g')
plt.bar(x, canvs_g, align='center', color='g')
plt.show()
# plt.plot(x, canvs_r,'r')
plt.bar(x, canvs_b, align='center', color='r')
plt.show()
