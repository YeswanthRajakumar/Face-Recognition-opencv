#Almost same as Detector
import os 
import cv2 as cv
import numpy as np
import sqlite3
from PIL import Image


classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
rec = cv.face.LBPHFaceRecognizer_create() 
font = cv.FONT_HERSHEY_SIMPLEX
rec.read("recognizer/trainingData.yml")
id =0

# for getting rows from DB

def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM people WHERE  ID ="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile



cap = cv.VideoCapture(1)
while True:
    ret,frame = cap.read()
    gray_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_img,scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        id,config = rec.predict(gray_img[y:y+h,x:x+w])
        profile = getProfile(id)

        if profile!=None:
            cv.putText(frame,"Id  : " + str(profile[0]),(x,y+h+30), font, 1, (200,0,0), 3, cv.LINE_AA)
            cv.putText(frame,"Name: " + str(profile[1]),(x,y+h+60), font, 1, (200,0,0), 3, cv.LINE_AA)
        if id == 0:
            cv.putText(frame,"Unknown",(x,y+h), font, 1, (200,0,0), 3, cv.LINE_AA)
            
    cv.imshow('Frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
       break
    
cap.release()
cv.destroyAllWindows()
    
