import keras
import cv2
from keras.models import load_model
import numpy as np

seq = load_model('D:\savedmodel')

ESC_VALUE = 27
EMOTION = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    
ok = True

while ok:        
    ret, frame = cap.read()
    frame2 = cv2.resize(frame[200:440, 120:360], None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    arr = np.asarray(gray).reshape(1, 48, 48, 1)
    y_pp = seq.predict(arr)
    i = np.argmax(y_pp[0])
    cv2.rectangle(frame, (200, 120), (440, 360), (0, 255, 0), 2)
    cv2.putText(frame, "Feeling: "+ EMOTION[i] + " - " + str(int(100 * y_pp[0, i])) + "%",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 0), 2)
    cv2.imshow('Input', frame)
    
    c = cv2.waitKey(1)
    if c == ESC_VALUE:
        break
        
cap.release()
cv2.destroyAllWindows()