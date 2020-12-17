import numpy as np
import cv2

def Sobel_edge_detection( f ):
	grad_x = cv2.Sobel( f, cv2.CV_32F, 1, 0, ksize = 3 )
	grad_y = cv2.Sobel( f, cv2.CV_32F, 0, 1, ksize = 3 )
	magnitude = abs( grad_x ) + abs( grad_y )
	g = np.uint8( np.clip( magnitude, 0, 255 ) )
	ret,g = cv2.threshold( g, 127, 255, 
		    cv2.THRESH_BINARY + cv2.THRESH_OTSU ) 
	return g
		
def main( ):
	img1 = cv2.imread( "Osaka.bmp", -1 )
	img2 = Sobel_edge_detection( img1 )
	cv2.imshow( "Original Image",  img1 )	
	cv2.imshow( "Sobel Edge Detection", img2 )
	cv2.waitKey( 0 )

main( )