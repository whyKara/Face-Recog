# import numpy as np
# import cv2
# faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0)
# cap.set(3,640) # set Width
# cap.set(4,480) # set Height
# while True:
#     ret, img = cap.read()
#     img = cv2.flip(img, -1)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(
#         gray,     
#         scaleFactor=1.2,
#         minNeighbors=5,     
#         minSize=(20, 20)
#     )
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]  
#     cv2.imshow('video',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27: # press 'ESC' to quit
#         break
# cap.release()
# cv2.destroyAllWindows()

import cv2
import face_recognition
import sqlite3

def enroll_face(name):
    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Enroll Mode - Press Esc to Exit', frame)

        # Detect faces
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            # Take the first face found
            encoding = face_recognition.face_encodings(frame, face_locations)[0]

            # Store the face encoding in the database
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO faces (name, encoding) VALUES (?, ?)', (name, str(encoding.tolist())))
            conn.commit()
            conn.close()

            break

        if cv2.waitKey(1) == 27:  # Esc key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
enroll_face('User1')

