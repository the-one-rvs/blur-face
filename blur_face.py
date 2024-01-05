import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0) #for webcam 0,1 refrence the first csam connected to computer,and so on...

while True:
    isTrue, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    haar_cascade = cv.CascadeClassifier("E:\Open CV\Project\Face_Blur\haar_face.xml")

    face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    faceimg = np.zeros((100,100),dtype='uint8')

    for (x,y,w,h) in face_rect:
        #cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), thickness = 2)
        faceimg = img[y:y+h,x:x+w]
        
        #gray_face = cv.cvtColor(faceimg, cv.COLOR_BGR2GRAY)
        n = 30
        while (n>0):
            faceimg=cv.GaussianBlur(faceimg,(31,31),0)
            n=n-1

        img[y:y+h,x:x+w] = faceimg

    cv.imshow("DETECTED FACE", img)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

cap.release()
cv.destroyAllWindows()