import numpy as np
import cv2

def Morphology_dilate(img):
    image = img.copy()
    height = img.shape[0]
    weight = img.shape[1]
    # 8-connected kernel
    kernel = np.array(((255, 255, 255),(255, 255, 255),(255, 255, 255)), dtype=np.int)
    
    for x in range(1,height - 1):
        for y in range(1,weight - 1):
            edge = img[x - 1:x + 2,y - 1:y + 2]
            edge = np.bitwise_and(edge,kernel)
            image[x - 1,y - 1] = np.max(edge)
    return image

def Morphology_erode(img):
    image = img.copy()
    height = img.shape[0]
    weight = img.shape[1]
    # 8-connected kernel
    kernel = np.array(((255, 255, 255),(255, 255, 255),(255, 255, 255)), dtype=np.int)
    
    for x in range(1,height - 1):
        for y in range(1,weight - 1):
            edge = img[x - 1:x + 2,y - 1:y + 2]
            edge = np.bitwise_and(edge,kernel)
            image[x - 1,y - 1] = np.min(edge)
    return image

def Morphology_open(img):
    img = Morphology_erode(img)
    img = Morphology_dilate(img)
    return img

def Morphology_close(img):
    img = Morphology_dilate(img)
    img = Morphology_erode(img)
    return img

img = cv2.imread("finger81.jpg", -1)

weight = img.shape[0]
height = img.shape[1]

hsv = np.zeros_like(img)

for x in range(weight):
    for y in range(height):
        b, g, r = img[x, y]

# b, g, r [0,1]
        b = b / 255
        g = g / 255
        r = r / 255

        MAX = max(r,g,b)
        MIN = min(r,g,b)
        # H[0..360]
        if MAX == MIN:  h = 0
        elif MAX == r and g >= b:   h = 60 * (g - b) / (MAX - MIN)
        elif MAX == r and g < b:    h = 60 * (g - b) / (MAX - MIN) + 360
        elif MAX == g:  h = 60 * (b - r) / (MAX - MIN) + 120
        elif MAX == b:  h = 60 * (r - g) / (MAX - MIN) + 240

        # s[0,1]
        if MAX == 0:    s = 0
        else:   s = 1 - MIN / MAX
        
        # v[,,1]
        v = MAX

        h = h / 360 * 180
        s = s * 255
        v = v * 255
        hsv[x,y] = np.array([h,s,v])

weight = hsv.shape[1]
height = hsv.shape[0]

hsv_catch = hsv.copy( )


for x in range(height):
    for y in range(weight):
        H = hsv[x,y,0] * 2
        S = hsv[x,y,1] / 255 * 100
        V = hsv[x,y,2] / 255 * 100
        if not ( H >= 0 and H <= 40 and S >= 30 and S <= 100 and V >= 30 and V <= 100 ):
            hsv_catch[x,y,0] = hsv_catch[x,y,1] = hsv_catch[x,y,2] = 0
        else:
            hsv_catch[x,y,0] = hsv_catch[x,y,1] = hsv_catch[x,y,2] = 255

cv2.imshow("img",img)
cv2.imshow("hsv",hsv)
cv2.imshow("hsv_catch",hsv_catch)
cv2.waitKey(0)

gray = cv2.imwrite("hsv_catch.jpg",hsv_catch)
hsv_catch = cv2.imread("hsv_catch.jpg",0)

hsv_catch_close = Morphology_close(hsv_catch)
hsv_catch_open = Morphology_open(hsv_catch)
cv2.imshow('hsv_catch_close',hsv_catch_close)
cv2.imshow('hsv_catch_open',hsv_catch_open)

cv2.waitKey(0)
