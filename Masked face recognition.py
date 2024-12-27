import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model-facemask.h5') 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        resized_roi = cv2.resize(face_roi, (224, 224))
        rgb_roi = cv2.cvtColor(resized_roi, cv2.COLOR_GRAY2RGB)
        normalized_roi = rgb_roi / 255.0
        reshaped_roi = np.reshape(normalized_roi, (1, 224, 224, 3))
                
        prediction = model.predict(reshaped_roi)
        mask_label = "With Mask" if prediction[0][0] > 0.5 else "Without Mask"
        
        color = (0, 255, 0) if mask_label == "With Mask" else (0, 0, 255)
        label = f"Face: {mask_label}"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = detect_mask(frame)
    cv2.imshow('Face Mask Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
