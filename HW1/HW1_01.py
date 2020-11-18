import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imshow(winname,mat) 顯示影像
cv2.imshow("gray", img)
# cv2.Waitkey([delay]) 等待按鍵
cv2.waitKey()

# 建立一維256位置的0陣列
canvs = np.zeros((256), np.int)
# img 像素維度 -> 一維 reshape(-1)
image = np.reshape(img, (-1))
# 計算向素值的數量
for num in image:
    canvs[num] += 1
# print(canvs)
# 直方圖
x = range(256)
plt.xlabel('gray Level')
plt.ylabel("number of poxels")
# plt.plot(x, canvs)
plt.bar(x, canvs, align='center', color='b')
plt.show()

""" 使用 calcHist 套件
# 計算直方圖每個 bin 的數值
# cv2.calcHist(影像, 通道, 遮罩, 區間數量, 數值範圍)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
"""

