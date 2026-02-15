import cv2
import face_recognition
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Path to dataset
path = 'dataset'
images = []
student_info = []

# Load images from dataset folder
for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    images.append(img)
    
    # Extract roll and name
    file_name = os.path.splitext(file)[0]
    roll, name = file_name.split("_")
    student_info.append((roll, name))

print("Dataset Loaded Successfully!")

# Function to encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding Faces...")
encodeListKnown = findEncodings(images)
print("Encoding Complete!")

def markAttendance(roll, name):
    date = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%H:%M:%S')
    
    filename = f'attendance/{date}.csv'
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=["Roll No", "Name", "Time"])
    
    # Avoid duplicate marking
    if roll not in df["Roll No"].astype(str).values:
        new_entry = {"Roll No": roll, "Name": name, "Time": time_now}
        
        # ✅ FIXED LINE (no append)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        
        df.to_csv(filename, index=False)
        print(f"{name} Marked Present")
    else:
        print(f"{name} Already Marked")

# Start Webcam
cap = cv2.VideoCapture(0)

print("Press Q to Exit")

while True:
    success, img = cap.read()
    
    imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            roll, name = student_info[matchIndex]
            
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"{roll}_{name}", (x1,y2+25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            markAttendance(roll, name)
    
    cv2.imshow('Automatic Attendance System', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
