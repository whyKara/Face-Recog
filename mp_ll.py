import mediapipe as mp
import face_recognition
import cv2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

import numpy as np

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.7) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Ignoring empty camera frame.")
            continue
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        if results.face_landmarks:
            # Convert the landmark points to numpy array
            face_landmarks = np.array([[p.x, p.y, p.z] for p in results.face_landmarks.landmark])
            
            # Calculate the difference in y-coordinates of the eyes to check for liveness
            left_eye_y = (face_landmarks[159][1] + face_landmarks[145][1]) / 2
            right_eye_y = (face_landmarks[386][1] + face_landmarks[374][1]) / 2
            eye_diff = np.abs(left_eye_y - right_eye_y)
            
            # Calculate the difference in x-coordinates of the shoulders to check for liveness
            left_shoulder_x = face_landmarks[11][0]
            right_shoulder_x = face_landmarks[12][0]
            shoulder_diff = np.abs(left_shoulder_x - right_shoulder_x)
            
            # Check if the differences are within a reasonable range to determine liveness
            if eye_diff < 0.05 and shoulder_diff > 0.0001:
                # Check for screen reflection in the eyes to detect phone screens or printed photos
                left_eye_reflection = face_landmarks[159][2] + face_landmarks[145][2]
                print(left_eye_reflection)
                right_eye_reflection = face_landmarks[386][2] + face_landmarks[374][2]
                print(right_eye_reflection)
                if left_eye_reflection > -0.05 and right_eye_reflection > -0.05:
                    print("Live face detected.")
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                             )
                else:
                    print("Liveness check failed. Face might not be live or it's a photo.")
            else:
                print("Liveness check failed. Face might not be live.")
        else:
            print("No face landmarks detected.")
        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
