from cv2 import cv2
import numpy as np
from scipy import signal


img = cv2.imread('house.jpg',0)
# 簽名
signature = cv2.imread('sign.png',0)


blur_img = cv2.GaussianBlur(img,(3,3),0)


gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)
magnitude = np.abs(gx) + np.abs(gy)


theta = np.degrees(np.arctan2(gy,gx))


height = magnitude.shape[0]
weight = magnitude.shape[1]
edge = np.zeros_like(magnitude)

for i in range(1,height - 1):
    for j in range(1,weight - 1):
        if ( ( (theta[i,j] >= -22.5) and (theta[i,j]< 22.5) ) or
                ( (theta[i,j] <= -157.5) and (theta[i,j] >= -180) ) or
                ( (theta[i,j] >= 157.5) and (theta[i,j] < 180) ) ):
            magnitude_max = max(magnitude[i, j - 1], magnitude[i, j + 1])
        elif (    ( (theta[i,j] >= 22.5) and (theta[i,j]< 67.5) ) or
                ( (theta[i,j] <= -112.5) and (theta[i,j] >= -157.5) ) ):
            magnitude_max = max(magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])
        elif (    ( (theta[i,j] >= 67.5) and (theta[i,j]< 112.5) ) or
                ( (theta[i,j] <= -67.5) and (theta[i,j] >= -112.5) ) ):
            magnitude_max = max(magnitude[i - 1, j], magnitude[i + 1, j])
        elif (    ( (theta[i,j] >= 112.5) and (theta[i,j]< 157.5) ) or
                ( (theta[i,j] <= -22.5) and (theta[i,j] >= -67.5) ) ):
            magnitude_max = max(magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])
        
        if magnitude[i, j] > magnitude_max:
            edge[i, j] = magnitude[i, j]

# 找有和強像素相連的弱像素
# canvas = connected(edge,100,200)

height = edge.shape[0]
weight = edge.shape[1]
# 加果是強像素設成2、弱像素設1、其他設0
for i in range(height):
    for j in range(weight):
        if edge[i,j] >= 200:
            edge[i,j] = 2
        elif 200 > edge[i,j] >= 100:
            edge[i,j] = 1
        else:
            edge[i,j] = 0

for i in range(1,height - 1):
    for j in range(1,weight - 1):
        if edge[i,j] == 2:
            dx = [-1,-1,-1,0,0,1,1,1]
            dy = [-1,0,1,-1,1,-1,0,1]

            for x,y in zip(dx,dy):
                if edge[i + x,j + y] == 1:
                    
                    edge[i + x,j + y] = 2

edge = np.where(edge == 2,255,0)
canvas = np.uint8(edge)

# 貼簽名
# paste_signature(canvas,signature)
height = canvas.shape[0]
weight = canvas.shape[1]
# signature = resize(signature)
canvas[height - 125:,weight - 200:] = cv2.bitwise_or(canvas[height - 125:,weight - 200:],signature)

import math

# img1 = cv2.imread( "Traffic_Lanes.bmp", -1 )
img2 = canvas.copy( )
# gray = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )
# edges = cv2.Canny( gray, 50, 200 )
# edges_HoughLinesP = canvas.copy()
# lines = cv2.HoughLineP ( canvas, 1, math.pi/180.0, 120 )
# for line in [l[0] for l in lines]:  # 畫線
#     leftx, boty, rightx, topy = line
#     cv2.line(edges_HoughLinesP, (leftx, boty), (rightx,topy), (255, 255, 0), 2)

lines = cv2.HoughLines( canvas, 1, math.pi/180.0, 120 )
if lines is not None:
	a,b,c = lines.shape
	for i in range( a ):
		rho = lines[i][0][0] #第一个元素是距离rho
		theta= lines[i][0][1] #第二个元素是角度theta
		if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线
			#该直线与第一行的交点
			pt1 = (int(rho/np.cos(theta)),0)
			#该直线与最后一行的焦点
			pt2 = (int((rho-img2.shape[0]*np.sin(theta))/np.cos(theta)),img2.shape[0])
			#绘制一条白线
			cv2.line( img2, pt1, pt2, (255))
		else: #水平直线
			# 该直线与第一列的交点
			pt1 = (0,int(rho/np.sin(theta)))
			#该直线与最后一列的交点
			pt2 = (img2.shape[1], int((rho-img2.shape[1]*np.cos(theta))/np.sin(theta)))
			#绘制一条直线
			cv2.line(img2, pt1, pt2, (255), 1)


cv2.imshow( "Hough Line Detection", img2 )
# cv2.imshow( "Hough Line Detection", edges_HoughLinesP)

cv2.waitKey( 0 )

cv2.imshow('canny',canvas)


cv2.waitKey(0)
cv2.destroyAllWindows()