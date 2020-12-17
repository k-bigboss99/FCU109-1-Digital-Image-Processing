import numpy as np
import cv2
import math

img1 = cv2.imread( "picture1.jpg", 0 )
img1 = cv2.GaussianBlur(img1,(3,3),0)
img2 = img1.copy( )

edges = cv2.Canny( img1, 50, 200 )
lines = cv2.HoughLines( edges, 1, math.pi/180.0, 135 )
if lines is not None:
	a,b,c = lines.shape
	for i in range( a ):
		rho = lines[i][0][0]
		theta = lines[i][0][1]
		a = math.cos( theta )
		b = math.sin( theta )
		x0, y0 = a*rho, b*rho
		pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
		pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
		cv2.line( img2, pt1, pt2, ( 255, 0, 0 ), 1)
# cv2.imshow( "Original Image", img1 )	
# cv2.imshow( "Canny Edge Detection", edges )
cv2.imshow( "Hough Line Detection", img2 )
cv2.waitKey( 0 )