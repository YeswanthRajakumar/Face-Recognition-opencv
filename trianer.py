import os
import cv2 as cv
import numpy as np
from PIL import Image

recognizer = cv.face.LBPHFaceRecognizer_create() 
path = 'Datasets'

def getImageWithId(path):
    imagepaths = [os.path.join(path,f) for f in os.listdir(path)]
    #print(imagepaths)
    faces =[]
    IDs =[]
    for imagepath in imagepaths:
        #converting it to PIL image
        faceImage = Image.open(imagepath).convert('L')
        #converting it to numpy array
        faceNp =np.array(faceImage,'uint8')

        #getting userid from name of picture
        ID = int(os.path.split(imagepath)[-1].split('.')[1])

        #append the values to the lists
        faces.append(faceNp)
        IDs.append(ID)
        
        cv.imshow("training",faceNp)
        cv.waitKey(20)

    return np.array(IDs),faces

IDs,faces = getImageWithId(path)

#using the train attribute from recognizer 
recognizer.train(faces,IDs)

recognizer.save('recognizer/trainingData.yml')
cv.destroyAllWindows()
