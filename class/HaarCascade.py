import numpy as np
import cv2
import time

# 讀取預先建立好的Haar Cascade 模型
face_cascade = cv2.CascadeClassifier('D:\FCU 109-1\opencv-4.4.0\data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('D:\FCU 109-1\opencv-4.4.0\data\haarcascades\haarcascade_eye.xml')

sTime = time.time()
# 讀取圖檔並轉成黑白圖(供haar cascade使用)
img = cv2.imread('Lenna.jpg')
img_cover = cv2.imread('effect.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#偵測臉部
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#對於每一個找到的臉部
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # 在臉的範圍之內搜尋眼睛的位置
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
eTime = time.time()

# 輸出並結束程式
print("執行完畢，費時"+str(eTime-sTime)+"秒")
cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()