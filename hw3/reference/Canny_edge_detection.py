import numpy as np
import cv2

img1 = cv2.imread( "picture1.jpg", 0 )
img2 = cv2.Canny( img1, 50, 200 )
cv2.imshow( "Original Image", img1 )	
# cv2.imshow( "Canny Edge Detection", img2 )
img3=cv2.imread("sign.png",0)
height = img2.shape[0]
weight = img2.shape[1]
img2[0:125,0:200] = cv2.bitwise_or(img2[0:125,0:200],img3)
cv2.imshow("test",img2)
cv2.waitKey( 0 )