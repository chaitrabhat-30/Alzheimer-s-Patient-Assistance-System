import cv2
import pytesseract
import streamlit as st
from PIL import Image
import numpy as np
from gtts import gTTS
import tempfile
import os
import time

# Path to tesseract.exe (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("📷 Live OCR with Camera + 🔊 Text-to-Speech")

# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Could not open camera")
else:
    frame_placeholder = st.empty()
    text_placeholder = st.empty()
    audio_placeholder = st.empty()

    last_text = ""  # store last spoken text

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # OCR
        text = pytesseract.image_to_string(gray, lang="eng").strip()

        # Show frame in browser
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Only update when new text is found
        if text and text != last_text:
            last_text = text  # update memory

            # Show extracted text
            text_placeholder.text_area("Extracted Text", text, height=150)

            # Convert text to speech and save temporary mp3
            tts = gTTS(text=text, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
                tts.save(tmpfile.name)
                audio_file = tmpfile.name

            # Play audio in Streamlit
            audio_placeholder.audio(audio_file)

        time.sleep(1)  # avoid rapid refresh
