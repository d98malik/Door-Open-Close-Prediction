import cv2 as cv 
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/second_model.h5")

cap = cv.VideoCapture(r"data\Test Video.mp4")

frame_count = 0
while True: 
    ret, frame = cap.read()
    frame_resized = cv.resize(frame, (224, 224))
    arr = np.expand_dims(frame_resized, axis=0)
    prediction = model.predict(arr)
    if ret == True:
        if prediction[0]<0.5:
            cv.putText(frame, "Closed", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv.imshow("frame", frame)
        
        else:
            cv.putText(frame, "Open", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv.imshow("frame", frame)
    frame_count+=1
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

