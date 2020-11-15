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

weight = img.shape[1]
height = img.shape[0]


for x in range(height):
    for y in range(weight):
        b, g, r = img[x, y]
        if not ( b >= 0 and b <= 255 and g >= 0 and g <= 212 and r >= 130 and r <= 230 ):
            img[x,y,0] = img[x,y,1] = img[x,y,2] = 0
        else:
            img[x,y,0] = img[x,y,1] = img[x,y,2] = 255
   


cv2.imshow("rgb_img",img)
cv2.waitKey(0)

rgb_img_close = Morphology_close(img)
rgb_img_open = Morphology_open(img)
cv2.imshow('rgb_img_close',rgb_img_close)
cv2.imshow('rgb_img_open',rgb_img_open)

cv2.waitKey(0)
