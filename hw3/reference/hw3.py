# import numpy as np
# import cv2

# def Sobel_edge_detection( f ):
# 	grad_x = cv2.Sobel( f, cv2.CV_32F, 1, 0, ksize = 3 )
# 	grad_y = cv2.Sobel( f, cv2.CV_32F, 0, 1, ksize = 3 )
# 	magnitude = abs( grad_x ) + abs( grad_y )
# 	magnitude = np.uint8( np.clip( magnitude, 0, 255 ) )
# 	ret,magnitude = cv2.threshold( magnitude, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU ) 
# 	return magnitude
		
# def main( ):
# 	magnitude = cv2.imread( "house.jpg", 0 )
# 	img2 = Sobel_edge_detection( magnitude )
# 	cv2.imshow( "Original Image",  magnitude )	
# 	cv2.imshow( "Sobel Edge Detection", img2 )
# 	cv2.waitKey( 0 )

# main( )

# magnitude = cv2.imread( "house.jpg", 0 )
# img2 = cv2.Canny( magnitude, 50, 200 )
# cv2.imshow( "Original Image", magnitude )	
# cv2.imshow( "Canny Edge Detection", img2 )
# cv2.waitKey( 0 )


# # # magnitude = cv2.imread( "house.jpg", 0 )

# # # gx = cv2.Sobel(magnitude, cv2.CV_64F, 1, 0, ksize = 3)
# # # gy = cv2.Sobel(magnitude, cv2.CV_64F, 0, 1, ksize = 3)

# # # angle = np.degrees(np.arctan2(gy,gx))
# # # angle[angle < 0] = angle[angle < 0] + 180
# # # print(angle)

# # # gx = cv2.convertScaleAbs(gx)
# # # gy = cv2.convertScaleAbs(gy)
# # # magnitude = cv2.addWeighted(gx,1,gy,1,0)

# # # cv2.imshow( "mm1", magnitude )
# # # cv2.waitKey( 0 )

# # # height = magnitude.shape[0]
# # # weight = magnitude.shape[1]
# # # edge = np.zeros(height, weight)

# # # for i in range(1,height - 1):
# # # 	for j in range(1,weight - 1):
# # # 		if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
# # # 			magnitude_max = max(magnitude[i, j - 1], magnitude[i, j + 1])
# # # 		elif (22.5 <= angle[i, j] < 67.5):
# # # 			magnitude_max = max(magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])
# # # 		elif (67.5 <= angle[i, j] < 112.5):
# # # 			magnitude_max = max(magnitude[i - 1, j], magnitude[i + 1, j])
# # # 		elif (112.5 <= angle[i, j] < 157.5):
# # # 			magnitude_max = max(magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])
		
# # # 		if magnitude[i, j] > magnitude_max:
# # # 			edge[i, j] = magnitude[i, j]
# # import cv2
# # import numpy as np

# # m1 = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
# # m2 = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))

# # # 第一步：完成高斯平滑滤波
# # img = cv2.imread("house.jpg",0)
# # img = cv2.GaussianBlur(img,(3,3),0)

# # # 第二步：完成一阶有限差分计算，计算每一点的梯度幅值与方向
# # magnitude = np.zeros(img.shape,dtype="uint8") # 与原图大小相同
# # angle = np.zeros(img.shape,dtype="float")  # 方向矩阵原图像大小
# # # img = cv2.copyMakeBorder(img,1,1,1,1,borderType=cv2.BORDER_REPLICATE)
# # height,weight = img.shape
# # for i in range(1,height-1):
# #     for j in range(1,weight-1):
# #         # Gy
# #         Gy = (m1 * img[i - 1:i + 2, j - 1:j + 2])
# #         # Gx
# #         Gx = (m2 * img[i - 1:i + 2, j - 1:j + 2])
# #         magnitude = abs(Gx) + abs(Gy)
# #         angle = np.degrees(np.arctan2(Gy,Gx))
# #         # if Gx[0] == 0:
# #         #     angle[i-1,j-1] = 90
# #         #     continue
# #         # else:
# #         #     temp = (np.arctan(Gy[0] / Gx[0]) ) * 180 / np.pi
# #         # if Gx[0]*Gy[0] > 0:
# #         #     if Gx[0] > 0:
# #         #         angle[i-1,j-1] = np.abs(temp)
# #         #     else:
# #         #         angle[i-1,j-1] = (np.abs(temp) - 180)
# #         # if Gx[0] * Gy[0] < 0:
# #         #     if Gx[0] > 0:
# #         #         angle[i-1,j-1] = (-1) * np.abs(temp)
# #         #     else:
# #         #         angle[i-1,j-1] = 180 - np.abs(temp)

        
# # for i in range(1,height - 2):
# #     for j in range(1, weight - 2):
# #         if (    ( (angle[i,j] >= -22.5) and (angle[i,j]< 22.5) ) or
# #                 ( (angle[i,j] <= -157.5) and (angle[i,j] >= -180) ) or
# #                 ( (angle[i,j] >= 157.5) and (angle[i,j] < 180) ) ):
# #             angle[i,j] = 0.0
# #         elif (    ( (angle[i,j] >= 22.5) and (angle[i,j]< 67.5) ) or
# #                 ( (angle[i,j] <= -112.5) and (angle[i,j] >= -157.5) ) ):
# #             angle[i,j] = 45.0
# #         elif (    ( (angle[i,j] >= 67.5) and (angle[i,j]< 112.5) ) or
# #                 ( (angle[i,j] <= -67.5) and (angle[i,j] >= -112.5) ) ):
# #             angle[i,j] = 90.0
# #         elif (    ( (angle[i,j] >= 112.5) and (angle[i,j]< 157.5) ) or
# #                 ( (angle[i,j] <= -22.5) and (angle[i,j] >= -67.5) ) ):
# #             angle[i,j] = -45.0

# # # 第三步：进行 非极大值抑制计算
# # img2 = np.zeros(magnitude.shape) # 非极大值抑制图像矩阵

# # for i in range(1,img2.shape[0]-1):
# #     for j in range(1,img2.shape[1]-1):
# #         if (angle[i,j] == 0.0) and (magnitude[i,j] == np.max([magnitude[i,j],magnitude[i+1,j],magnitude[i-1,j]]) ):
# #                 img2[i,j] = magnitude[i,j]

# #         if (angle[i,j] == -45.0) and magnitude[i,j] == np.max([magnitude[i,j],magnitude[i-1,j-1],magnitude[i+1,j+1]]):
# #                 img2[i,j] = magnitude[i,j]

# #         if (angle[i,j] == 90.0) and  magnitude[i,j] == np.max([magnitude[i,j],magnitude[i,j+1],magnitude[i,j-1]]):
# #                 img2[i,j] = magnitude[i,j]

# #         if (angle[i,j] == 45.0) and magnitude[i,j] == np.max([magnitude[i,j],magnitude[i-1,j+1],magnitude[i+1,j-1]]):
# #                 img2[i,j] = magnitude[i,j]

# # # 第四步：双阈值检测和边缘连接
# # img3 = np.zeros(img2.shape) #定义双阈值图像
# # # TL = 0.4*np.max(img2)
# # # TH = 0.5*np.max(img2)
# # TL = 50
# # TH = 100
# # #关键在这两个阈值的选择
# # for i in range(1,img3.shape[0]-1): 
# #     for j in range(1,img3.shape[1]-1):
# #         if img2[i,j] < TL:
# #             img3[i,j] = 0
# #         elif img2[i,j] > TH:
# #             img3[i,j] = 255
# #         elif (( img2[i+1,j] < TH) or (img2[i-1,j] < TH )or( img2[i,j+1] < TH )or
# #                 (img2[i,j-1] < TH) or (img2[i-1, j-1] < TH )or ( img2[i-1, j+1] < TH) or
# #                    ( img2[i+1, j+1] < TH ) or ( img2[i+1, j-1] < TH) ):
# #             img3[i,j] = 255


# # cv2.imshow("1",img)  		  # 原始图像
# # cv2.imshow("2",magnitude)       # 梯度幅值图
# # cv2.imshow("3",img2)       #非极大值抑制灰度图
# # cv2.imshow("4",img3)       # 最终效果图
# # cv2.imshow("angle",angle) #角度值灰度图
# # cv2.waitKey(0)

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("house.jpg",0)
img = cv2.GaussianBlur(img,(3,3),2)

magnitude = np.zeros(img.shape,dtype="uint8") # 与原图大小相同
angle = np.zeros(img.shape,dtype="float")  # 方向矩阵原图像大小
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)
magnitude = cv2.addWeighted(gx,1,gy,1,0)


height = magnitude.shape[0]
weight = magnitude.shape[1]

angle = np.degrees(np.arctan2(gy,gx))

for i in range(1,height - 2):
    for j in range(1, weight - 2):
        if (    ( (angle[i,j] >= -22.5) and (angle[i,j]< 22.5) ) or
                ( (angle[i,j] <= -157.5) and (angle[i,j] >= -180) ) or
                ( (angle[i,j] >= 157.5) and (angle[i,j] < 180) ) ):
            angle[i,j] = 0.0
        elif (    ( (angle[i,j] >= 22.5) and (angle[i,j]< 67.5) ) or
                ( (angle[i,j] <= -112.5) and (angle[i,j] >= -157.5) ) ):
            angle[i,j] = 45.0
        elif (    ( (angle[i,j] >= 67.5) and (angle[i,j]< 112.5) ) or
                ( (angle[i,j] <= -67.5) and (angle[i,j] >= -112.5) ) ):
            angle[i,j] = 90.0
        elif (    ( (angle[i,j] >= 112.5) and (angle[i,j]< 157.5) ) or
                ( (angle[i,j] <= -22.5) and (angle[i,j] >= -67.5) ) ):
            angle[i,j] = -45.0

# 第三步：进行 非极大值抑制计算
img2 = np.zeros(magnitude.shape) # 非极大值抑制图像矩阵

for i in range(1,img2.shape[0]-1):
    for j in range(1,img2.shape[1]-1):
        if (angle[i,j] == 0.0) and (magnitude[i,j] == np.max([magnitude[i,j],magnitude[i+1,j],magnitude[i-1,j]]) ):
                img2[i,j] = magnitude[i,j]

        if (angle[i,j] == -45.0) and magnitude[i,j] == np.max([magnitude[i,j],magnitude[i-1,j-1],magnitude[i+1,j+1]]):
                img2[i,j] = magnitude[i,j]

        if (angle[i,j] == 90.0) and  magnitude[i,j] == np.max([magnitude[i,j],magnitude[i,j+1],magnitude[i,j-1]]):
                img2[i,j] = magnitude[i,j]

        if (angle[i,j] == 45.0) and magnitude[i,j] == np.max([magnitude[i,j],magnitude[i-1,j+1],magnitude[i+1,j-1]]):
                img2[i,j] = magnitude[i,j]

# 第四步：双阈值检测和边缘连接
img3 = np.zeros(img2.shape) #定义双阈值图像
# TL = 0.4*np.max(img2)
# TH = 0.5*np.max(img2)
TL = 50
TH = 100
#关键在这两个阈值的选择
for i in range(1,img3.shape[0]-1): 
    for j in range(1,img3.shape[1]-1):
        if img2[i,j] < TL:
            img3[i,j] = 0
        elif img2[i,j] > TH:
            img3[i,j] = 255
        elif (( img2[i+1,j] < TH) or (img2[i-1,j] < TH )or( img2[i,j+1] < TH )or
                (img2[i,j-1] < TH) or (img2[i-1, j-1] < TH )or ( img2[i-1, j+1] < TH) or
                   ( img2[i+1, j+1] < TH ) or ( img2[i+1, j-1] < TH) ):
            img3[i,j] = 255


cv2.imshow("1",img)  		  # 原始图像
cv2.imshow("2",magnitude)       # 梯度幅值图
cv2.imshow("3",img2)       #非极大值抑制灰度图
cv2.imshow("4",img3)       # 最终效果图
cv2.imshow("angle",angle) #角度值灰度图
cv2.waitKey(0)