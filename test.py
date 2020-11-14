import numpy as np
import cv2

img = cv2.imread("finger2.jpg", -1)

# weight = img.shape[0]//5 #取整數
# height = img.shape[1]//5

# img = cv2.resize(img, (height, weight))
# cv2.imshow("after",img)
# cv2.waitKey(0)
# hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
# cv2.imshow( "Original Image", img )
# cv2.imshow( "HSV Color Segmentation", hsv )
# cv2.waitKey( 0 )
'''
def bgr_to_hsv(img):
    canvas = np.zeros_like(img,np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            canvas[y,x] = get_hsv(img[y,x])
    return canvas
'''

weight = img.shape[0]
height = img.shape[1]

for x in range(weight):
    for y in range(height):
        b, g, r = img[x, y]
# print(b, g, r)

# h = s = v = np.float32(0)
# value = np.float32(value) / 255.0
# b,g,r = value

# b, g, r [0,1]
b = b / 255
g = g / 255
r = r / 255
# print(b, g, r)

MAX = max(r,g,b)
MIN = min(r,g,b)
# H[0..360]
if MAX == MIN:
    h = 0
elif MAX == r and g >= b:
    h = 60 * (g - b) / (MAX - MIN)
elif MAX == r and g < b:
    h = 60 * (g - b) / (MAX - MIN) + 360
elif MAX == g:
    h = 60 * (b - r) / (MAX - MIN) + 120
elif MAX == b:
    h = 60 * (r - g) / (MAX - MIN) + 240
# print(h)
# s[0,1]
if MAX == 0:
    s = 0
else:
    s = 1 - MIN / MAX
# v[,,1]
v = MAX

h = h / 360 * 180
s = s * 255
v = v * 255
# print(h,s,v)

g = img.copy( )
# cv2.imshow("g",g)
# cv2.waitKey(0)
nr, nc = img.shape[:2]
# print(img.shape[:2])
# hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )

for i in range( nr ):
    for j in range( nc ):
        H = h * 2
        S = s / 255 * 100
        V = v / 255 * 100
        if not ( H >= 30 and H <= 70 and S >= 30 and S <= 100 and V >= 30 and V <= 100 ):
            g[i,j,0] = g[i,j,1] = g[i,j,2] = 0
print(H,S,V)
# cv2.imshow("after2",g)
# cv2.waitKey(0)