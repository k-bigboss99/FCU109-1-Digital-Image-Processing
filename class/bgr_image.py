import numpy as np
import cv2

# 灰階影像
# img = cv2.imread("1.jpg",0)
# RGB 通道影像
img = cv2.imread("sample.bmp",-1)
cv2.imshow("img",img)
cv2.waitKey()

color_B = np.zeros((549,1024,3))
color_G = np.zeros((549,1024,3))
color_R = np.zeros((549,1024,3))

color_B[:,:,0] = img[:,:,0]
color_G[:,:,1] = img[:,:,1]
color_R[:,:,2] = img[:,:,2]
cv2.imshow("img_B",color_B)
cv2.imshow("img_G",color_G)
cv2.imshow("img_R",color_R)
cv2.waitKey()

"""
row,column,channel = img.shape
print(type(img))
print("number of rows = ",row)
print("number of column = ",column)
print("number of channel = ",channel)
print(img)
"""
