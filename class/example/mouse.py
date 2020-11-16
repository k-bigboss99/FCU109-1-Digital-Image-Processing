import cv2

global img

def onMouse(event,x,y,flags,param):
    x,y = y,x
    if img.ndim != 3:
        print("(x,y)=%d,%d" %(x,y),end=" ")
        print("Gray-Level = %3d" %img[x,y])

    else:
        print("(x,y)=%d,%d" %(x,y),end=" ")
        print("R,G,B = (%3d,%3d,%3d)" %(img[x,y,2],img[x,y,1],img[x,y,0]))

img = cv2.imread("sample.jpg",-1)
# cv2.namedWindow("onMouse")
# cv2.setMouseCallback("onMouse",onMouse)
# cv2.imshow("onMouse",img)
cv2.imshow("sample_resize",cv2.resize(img,(500,250)))
cv2.waitKey()
