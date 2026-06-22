import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image

# ----------------- Paths -----------------
dataset_path = "dataset"
trainer_path = "trainer.yml"
os.makedirs(dataset_path, exist_ok=True)

# Haar cascade and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


# ----------------- Functions -----------------
def capture_faces(user_id, num_samples=50):
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0
    st.info("Starting face capture. Look at the camera...")

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to open camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{dataset_path}/User.{user_id}.{count}.jpg", face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Capturing {count}/{num_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_samples:
            break

    cam.release()
    cv2.destroyAllWindows()
    st.success(f"{count} face samples captured for User {user_id}")


def train_model():
    st.info("Training model...")
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    face_samples, ids = [], []

    for image_path in image_paths:
        gray_img = Image.open(image_path).convert("L")
        img_np = np.array(gray_img, "uint8")
        user_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = face_cascade.detectMultiScale(img_np)

        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(user_id)

    recognizer.train(face_samples, np.array(ids))
    recognizer.save(trainer_path)
    st.success("Model trained and saved.")


def recognize_faces():
    st.info("Starting real-time recognition. Press Q to quit.")
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not os.path.exists(trainer_path):
        st.error("No trained model found. Train first.")
        return

    recognizer.read(trainer_path)

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to open camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 60:
                label = f"User {id_}"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} - {round(100 - confidence)}%", (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# ----------------- Streamlit UI -----------------
st.title("Face Recognition System")

role = st.sidebar.radio("Select Role", ["Admin", "User"])

if role == "Admin":
    st.subheader("Admin Panel")

    if st.button("Face Registration"):
        user_id = st.text_input("Enter User ID (number):", "")
        if user_id:
            capture_faces(user_id)
            train_model()
            recognize_faces()
        else:
            st.warning("Enter User ID before proceeding.")

elif role == "User":
    st.subheader("User Panel")
    if st.button("Start Recognition"):
        recognize_faces()
