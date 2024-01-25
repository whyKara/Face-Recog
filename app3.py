import cv2
import face_recognition
import sqlite3

def recognize_faces():
    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Recognition Mode - Press Esc to Exit', frame)

        # Detect faces
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if face_encodings:
            for encoding in face_encodings:
                # Compare with stored encodings
                conn = sqlite3.connect('attendance.db')
                cursor = conn.cursor()
                cursor.execute('SELECT id, name, encoding FROM faces')
                records = cursor.fetchall()

                for record in records:
                    stored_encoding = face_recognition.face_encodings([eval(record[2])])[0]
                    match = face_recognition.compare_faces([stored_encoding], encoding)[0]

                    if match:
                        # Update attendance in the database
                        cursor.execute('UPDATE faces SET is_present = 1 WHERE id = ?', (record[0],))
                        conn.commit()
                        break

                conn.close()

        if cv2.waitKey(1) == 27:  # Esc key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
recognize_faces()
