import numpy as np
import cv2

img1 = cv2.imread( "Osaka.bmp", -1 )
img2 = cv2.Canny( img1, 50, 200 )
cv2.imshow( "Original Image", img1 )	
cv2.imshow( "Canny Edge Detection", img2 )
cv2.waitKey( 0 )