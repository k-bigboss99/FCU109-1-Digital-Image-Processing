import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram( f ):
	hist = cv2.calcHist( [f], [0], None, [256], [0,256] )
	plt.plot( hist )
	plt.xlim( [0,256] )
	plt.xlabel( "Intensity" )
	plt.ylabel( "#Intensities" )
	plt.show( )

def main( ):
	img = cv2.imread( "cat.jpg", 0 )
	img2 = cv2.equalizeHist( img )
	cv2.imshow( "Original Image", img )
	histogram( img )
	cv2.waitKey( 0 )

	cv2.imshow( "Histogram Equalization", img2 )
	histogram( img2 )
	cv2.waitKey( 0 )

main( )