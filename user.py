import cv2
import os
import numpy as np

# Paths
trainer_path = "trainer.yml"

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Check if trainer exists
if not os.path.exists(trainer_path):
    print("[ERROR] Trainer file not found! Please train first using the admin script.")
    exit()

# Load trained model
recognizer.read(trainer_path)

def recognize_faces():
    print("[INFO] Starting real-time face recognition...")
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 60:  # Lower confidence = better match
                label = f"User {id_}"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} - {round(100 - confidence)}%", (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_faces()
