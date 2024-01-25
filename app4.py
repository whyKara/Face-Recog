import cv2
import face_recognition
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

# Open camera
cap = cv2.VideoCapture(0)

def register_face():
    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Register Mode - Press Space to Capture, Esc to Exit', frame)

        k = cv2.waitKey(1)
        if k%256 == 27:  # Esc key to exit
            print("Escape hit, closing...")
            break
        elif k%256 == 32:  # Space key to save image
            # Detect face
            faces = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            if faces is ():
                print("No face detected, try again")
                continue

            for (x, y, w, h) in faces:
                roi_color = frame[y:y+h, x:x+w]
                # Save the captured image into the dataset folder
                cv2.imwrite("dataset/User." + str(id) + '.' + str(sampleNum) + ".jpg", roi_color)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                sampleNum += 1
            cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()

# Menu for registering faces
def menu():
    while True:
        print("\n [INFO] Enter 1 to register a new face, 0 to exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            register_face()
        elif choice == '0':
            break
        else:
            print("Invalid choice, please try again.")

menu()



while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)
        if confidence < 100:
            confidence = "  {0}%".format(round(100 - confidence))
            print("Face is real with confidence: " + confidence)
        else:
            print("Face is fake")

    cv2.imshow('Recognition Mode - Press Esc to Exit', frame)

    if cv2.waitKey(1) == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
