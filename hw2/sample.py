import numpy as np
import cv2

def HSV_color_segmentation( f, H1, H2, S1, S2, V1, V2 ):
	canvas = f.copy( )
	nr, nc = f.shape[:2]
	hsv = cv2.cvtColor( f, cv2.COLOR_BGR2HSV )
	cv2.imshow( "HSV Color Image", hsv )
	for x in range( nr ):
		for y in range( nc ):
			H = hsv[x,y,0] * 2
			S = hsv[x,y,1] / 255 * 100
			V = hsv[x,y,2] / 255 * 100
			if not ( H >= H1 and H <= H2 and S >= S1 and S <= S2 and V >= V1 and V <= V2 ):
				canvas[x,y,0] = canvas[x,y,1] = canvas[x,y,2] = 0
			else:
				canvas[x,y,0] = canvas[x,y,1] = canvas[x,y,2] = 255
	return canvas

def main( ):
	img1 = cv2.imread( "image1.jpg", -1 )
	img2 = HSV_color_segmentation( img1, 0, 40, 30, 100, 30, 100 )
	cv2.imshow( "Original Image", img1 )
	cv2.imshow( "HSV Color Segmentation", img2 )
	cv2.waitKey( 0 )

main( )