import numpy as np
import cv2
img = cv2.imread("sample.jpg",-1)
# BMP：無壓縮 
cv2.imwrite("sample.bmp",img)