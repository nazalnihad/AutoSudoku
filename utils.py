import cv2
import numpy as np
from tensorflow.keras.models import load_model

def setModel():
    model = load_model('resources/myModel.h5')
    return model

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,1,1,11,2)
    return imgThreshold

def reorder(points):
    points = points.reshape((4,2))
    newPoints = np.zeros((4,1,2),dtype=np.int32)
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points,axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]
    return newPoints

def largestContour(contours):
    largest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            if area > max_area and len(approx) == 4:
                largest = approx
                max_area = area
    
    return largest,max_area

def splitImg(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def getPredictions(boxes,model):
    result = []
    for box in boxes:
        img = np.asarray(box)
        img = img[4:img.shape[0]-4,4:img.shape[1]-4]
        img = cv2.resize(img,(28,28))
        img = img/255
        img = img.reshape(1,28,28,1)
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

a = setModel()
a.describe()