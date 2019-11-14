import cv2 as cv
import numpy as np
import sqlite3
    
cap = cv.VideoCapture(1)
classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#function for Db insertion and Updation
def insertOrUpdate(Id,Name):
    conn = sqlite3.connect("FaceBase.db")#db file name
    cmd = "SELECT * FROM people WHERE ID = "+str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
         isRecordExist=1
    if(isRecordExist==1):
        cmd = "UPDATE people SET Name ="+str(Name)+"WHERE ID ="+str(Id)
    else:
        cmd = "INSERT INTO people(ID,Name)values("+str(Id)+","+str(Name)+")"
    cursor = conn.execute(cmd)
    conn.commit()
    conn.close()


#Giving id and name for Captured Face

F_id = input("Enter the person  ID   : ")
F_name= input("Enter the person NAME : ")

insertOrUpdate(F_id,F_name)

SampleNo = 0
while True:
    ret,frame = cap.read()
    gray_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_img,1.2,5)
    for (x,y,w,h) in faces:
        #Whenever it Captures a face,sampleno increased by one  
        SampleNo = SampleNo+1
        #captured image is written in the folder
        cv.imwrite("Datasets/user."+str(F_id)+"."+str(SampleNo)+".jpg",gray_img[y:y+h,x:x+w])
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imshow('Frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
       break
    
cap.release()
cv.destroyAllWindows()
    
