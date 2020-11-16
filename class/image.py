import numpy as np
import cv2

# 建立屏幕
"""
# np.zeros 黑屏, np.ones 白幕
img1 = np.zeros([1024,1024], dtype="uint8")
img2 = np.zeros([1024,1024], dtype="uint8")
img3 = np.ones([512,512], dtype="float32")

cv2.imshow("test1", img1)
cv2.imshow("test2", img2)
cv2.imshow("test3", img3)
cv2.waitKey()
"""

# 5*5 black white
"""
img4 = np.zeros([500,500],dtype="uint8")
for i in range(5):
    for j in range(5):
        if((i+j)%2 == 1):
            for k in range(100):
                for l in range(100):
                    img4[i*100 + k][j*100 + l]=255
cv2.imshow('image', img4)
cv2.waitKey()
"""

# 影像大小調整 reshape，面積需相同
"""
img = cv2.imread("sample.jpg",0)
# img_size(549,1024)
image_reshape = img.reshape(1098,512)
cv2.imshow("image_reshape", image_reshape)
cv2.waitKey()
"""
# 建立視窗，中央藍屏 (b,g,r) = (255,0,0)
"""
img = np.zeros([500,500,3],dtype="uint8")
cv2.rectangle(img, (100,100), (200,200), (255,0,0), -1)
cv2.imshow("img", img)
cv2.waitKey()
"""

# 5*1 black white
"""
img = np.zeros([500,500],dtype="uint8")
for i in range(0, 5, 2):
    cv2.rectangle(img, (100*i,0), (100*i+100,100),255, -1)
cv2.imshow("img", img)
cv2.waitKey()
"""