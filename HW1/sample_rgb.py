import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram( f ):
	color = ( 'b', 'g', 'r' )
	for i, col in enumerate( color ):
		hist = cv2.calcHist( f, [i], None, [256], [0,256] )
		plt.plot( hist, color = col )
	plt.xlim( [0,256] )
	plt.xlabel( "Intensity" )
	plt.ylabel( "#Intensities" )
	plt.show( )

def RGB_histogram_equalization( f ):
	canvas = f.copy( )
	for k in range( 3 ):
		canvas[:,:,k] = cv2.equalizeHist( f[:,:,k] )
	return canvas

def main( ):
	img = cv2.imread( "cat.jpg", -1 )
	img2 = RGB_histogram_equalization( img )
	cv2.imshow( "Original_RGB Image", img )
	histogram( img )
	cv2.waitKey( 0 )

	cv2.imshow( "Histogram Equalization_RGB", img2 )
	histogram( img2 )
	cv2.waitKey( 0 )

main( )
