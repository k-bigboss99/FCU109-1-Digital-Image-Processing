import numpy as np
import cv2
from scipy import signal

def combine(img,c):
    i,j = c
    # dx、dy 分別是周圍8個點的位置
    dx = [-1,-1,-1,0,0,1,1,1]
    dy = [-1,0,1,-1,1,-1,0,1]

    for x,y in zip(dx,dy):
        if img[i + x,j + y] == 1:
            # 如果有弱像素存在把那個點設成強像素、再移動到那個點、用遞廻不斷查看下一個點
            img[i + x,j + y] = 2
            combine(img,(i + x,j + y))



img = cv2.imread('house.jpg',0)
signature = cv2.imread('signature.png',0)
img_blur = cv2.GaussianBlur(img,(3,3),0)
cv2.imshow( "img_blur", img_blur )
kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
gx = signal.convolve2d(img,kernel_x,'same','symm')
gy = signal.convolve2d(img,kernel_y,'same','symm')
magnitude = np.abs(gx) + np.abs(gy)

# 算梯度方向
angle = np.degrees(np.arctan2(gy,gx))
# 小於 0 的加上 180 讓它轉半圈
# 因為方向都在對角(差180)
angle[angle < 0] += 180

# gx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize = 3)
# gy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize = 3)
# angle = np.degrees(np.arctan2(gy,gx))
# angle[angle < 0] += 180

# gx = cv2.convertScaleAbs(gx)
# gy = cv2.convertScaleAbs(gy)
# magnitude = cv2.addWeighted(gx,1,gy,1,0)
# magnitude = abs(gx) + abs(gy)

# cv2.imshow("img_magnitude", magnitude)
# cv2.waitKey( 0 )

height = magnitude.shape[0]
weight = magnitude.shape[1]
border = np.zeros_like(magnitude)
for i in range(1, height- 1):
    for j in range(1,weight - 1):
        # 找當前點的梯度方向上的另外兩點
        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
            max_ = max(magnitude[i, j - 1], magnitude[i, j + 1])
        elif (22.5 <= angle[i, j] < 67.5):
            max_ = max(magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])
        elif (67.5 <= angle[i, j] < 112.5):
            max_ = max(magnitude[i - 1, j], magnitude[i + 1, j])
        elif (112.5 <= angle[i, j] < 157.5):
            max_ = max(magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])
        
        if magnitude[i, j] > max_:
            # 加果當前點是最大的就保留
            border[i, j] = magnitude[i, j]



# 找有和強像素相連的弱像素
# canvas = connected(border,100,200)
height = border.shape[0]
weight = border.shape[1]

# 加果是強像素設成2、弱像素設1、其他設0
for i in range(height):
    for j in range(weight):
        if border[i,j] >= 200:
            border[i,j] = 2
        elif 200 > border[i,j] >= 100:
            border[i,j] = 1
        else:
            border[i,j] = 0

for i in range(1,height - 1):
    for j in range(1,weight - 1):
        if border[i,j] == 2:
            # 確認強像素周圍的點
            combine(border,(i,j))
border = np.where(img == 2,255,0)
canvas = np.uint8(border)

# height = canvas.shape[0]
# weight = canvas.shape[1]
# canvas[height - 125:,weight - 200:] = cv2.bitwise_or(canvas[height - 125:,weight - 200:],signature)

cv2.imshow('canny',canvas)


cv2.waitKey(0)
cv2.destroyAllWindows()