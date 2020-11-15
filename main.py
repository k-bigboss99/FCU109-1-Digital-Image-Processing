from cv2 import cv2
import numpy as np
import os

def get_hsv(value):
    h = s = v = np.float32(0)
    value = np.float32(value) / 255.0
    b,g,r = value
    max_ = np.max(value)
    min_ = np.min(value)
    # h 
    if max_ == min_:
        h = 0
    elif max_ == r and g >= b:
        h = 60 * (g - b) / (max_ - min_)
    elif max_ == r and g < b:
        h = 60 * (g - b) / (max_ - min_) + 360
    elif max_ == g:
        h = 60 * (b - r) / (max_ - min_) + 120
    elif max_ == b:
        h = 60 * (r - g) / (max_ - min_) + 240
    # s
    if max_ == 0:
        s = 0
    else:
        s = 1 - min_ / max_
    # v
    v = max_
    h = np.uint8(h / 360 * 180)
    s = np.uint8(s * 255)
    v = np.uint8(v * 255)

    return np.array([h,s,v])

def bgr_to_hsv(img):
    canvas = np.zeros_like(img,np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            canvas[y,x] = get_hsv(img[y,x])
    return canvas

def click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(param[y,x])

def inRange(img,lower,upper):
    img = np.where((
        lower[0] <= img[...,0]) &\
        (img[...,0] <= upper[0]) &\
        (lower[1] <= img[...,1]) &\
        (img[...,1] <= upper[1]) &\
        (lower[2] <= img[...,2]) &\
        (img[...,2] <= upper[2]),
        255,
        0
    )
    img = np.uint8(img)
    return img

def dilate(img,iteration = 1):
    # canvas = np.zeros_like(img)
    # img = np.pad(img,1,mode = 'constant')
    # kernel = np.ones((3,3),np.uint8) * 255
    # for _ in range(iteration):
    #     for y in range(1,img.shape[0] - 1):
    #         for x in range(1,img.shape[1] - 1):
    #             region = img[y - kernel.shape[0] // 2:y + kernel.shape[0] // 2 + 1,x - kernel.shape[1] // 2:x + kernel.shape[1] // 2 + 1]
    #             region = np.bitwise_and(region,kernel)
    #             canvas[y - 1,x - 1] = np.max(region)
    # return canvas
    canvas = img.copy()
    # img = np.pad(img,1,mode = 'constant')
    H = img.shape[0]
    W = img.shape[1]
    kernel = np.array(((0, 255, 0),(255, 0, 255),(0, 255, 0)), dtype=np.int)
    for y in range(1,H - 1):
        for x in range(1,W - 1):
            region = img[y - 1:y + 2,x - 1:x + 2]
            region = np.bitwise_and(region,kernel)
            canvas[y - 1,x - 1] = np.max(region)
    return canvas

def erode(img,iteration = 1):
    canvas = np.zeros_like(img)
    img = np.pad(img,1,mode = 'constant')
    kernel = np.ones((3,3),np.uint8) * 255
    for _ in range(iteration):
        for y in range(1,img.shape[0] - 1):
            for x in range(1,img.shape[1] - 1):
                region = img[y - kernel.shape[0] // 2:y + kernel.shape[0] // 2 + 1,x - kernel.shape[1] // 2:x + kernel.shape[1] // 2 + 1]
                region = np.bitwise_and(region,kernel)
                canvas[y - 1,x - 1] = np.min(region)
    return canvas

def morph_open(img,iteration = 1):
    for _ in range(iteration):
        img = erode(img)
        img = dilate(img)
    return img

def morph_close(img,iteration = 1):
    for _ in range(iteration):
        img = dilate(img)
        img = erode(img)
    return img

img = cv2.imread('finger11.jpg')
img = cv2.resize(img,(img.shape[1],img.shape[0]))
hsv_img = bgr_to_hsv(img)
hsv_bin_img = inRange(hsv_img,[0,40,190],[50,150,255])
print(hsv_bin_img.shape)
hsv_bin_img1 = morph_close(hsv_bin_img,1)
hsv_bin_img2 = morph_open(hsv_bin_img,1)
# bgr_bin_img = inRange(img,[100,120,130],[150,170,230])
# bgr_bin_img = morph_open(bgr_bin_img,3)

# cv2.namedWindow('hsv')
# cv2.setMouseCallback('hsv',click,param = img)
# cv2.imshow('img',img)
cv2.imshow('hsv1',hsv_bin_img1)
cv2.imshow('hsv2',hsv_bin_img2)
# cv2.imshow('bgr',bgr_bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
