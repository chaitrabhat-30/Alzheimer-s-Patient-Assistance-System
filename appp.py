from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import subprocess
import time
import psutil
import signal
import cv2
import pytesseract
import numpy as np
from gtts import gTTS
from PIL import Image
import tempfile
import base64
import threading
import google.generativeai as genai
from datetime import datetime
import time
import requests
from shapely.geometry import Point, Polygon  # pip install shapely

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

# Path to your Streamlit app file
STREAMLIT_SCRIPT = "reminder_app.py"   # <-- replace with your streamlit file name

# Path to tesseract.exe (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Configure Gemini API - Replace with your actual API key
GEMINI_API_KEY = "Replace_with_your_Gemini_API_key"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash')

# GPS Tracking and Geofencing Configuration
TELEGRAM_BOT_TOKEN = 'replace_with_your_telegram_bot_token'
TELEGRAM_CHAT_ID = 'replace_with_your_telegram_chat_id'
NODEMCU_IP = 'replace_with_your_nodemcu_ip'  
URL = f"http://{NODEMCU_IP}/gps"
DB_NAME = "geofence.db"

# Initialize GPS and geofencing databases
def init_geofence_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS geofence (
                id INTEGER PRIMARY KEY,
                lat REAL,
                lng REAL
            )
        ''')

init_geofence_db()

@app.route('/chatbot')
def home():
    """Render the main chat interface"""
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and return Gemini API responses"""
    try:
        # Get user message from request
        user_message = request.json.get('question', '')
        
        if not user_message:
            return jsonify({'error': 'No question provided'}), 400
        
        # Generate response using Gemini API
        response = model.generate_content(user_message)
        
        return jsonify({
            'success': True,
            'answer': response.text
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing request: {str(e)}'
        }), 500

@app.route('/api/status')
def status():
    """Check API status"""
    try:
        # Simple test to verify API connection
        test_response = model.generate_content("Hello")
        return jsonify({'success': True})
    except:
        return jsonify({'success': False})
# Global variables for OCR functionality
last_text = ""
camera = None
is_camera_active = False
audio_file_path = None
current_mode = "image"  # Default mode

# Database initialization
def init_db():
    conn = sqlite3.connect('memoryaid.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL,
                 email TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 user_type TEXT DEFAULT 'user',
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()

# Face Recognition Setup
dataset_path = "dataset"
trainer_path = "trainer.yml"
os.makedirs(dataset_path, exist_ok=True)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
id_to_name = {
    1: "Neha",
    2: "Shreya",
    3: "Pallavi",
    4: "Ramya",
    5: "Anjana",
    6: "Pallavi"
}

# Create LBPH recognizer (requires opencv-contrib-python)
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except:
    recognizer = None
    print("Warning: OpenCV contrib module not found. Face recognition disabled.")

# Global variables for face recognition
face_capture_active = False
face_recognition_active = False
face_camera = None

def generate_frames():
    global last_text, camera, audio_file_path
    
    if camera is None or not camera.isOpened():
        return
    
    while is_camera_active:
        success, frame = camera.read()
        if not success:
            break
        
        # Preprocess for OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        gray = cv2.medianBlur(gray, 3)
        
        # OCR extraction
        text = pytesseract.image_to_string(gray, lang="eng", config='--psm 6').strip()
        print("Extracted:", text)
        
        if text and text != last_text:
            last_text = text
            try:
                tts = gTTS(text=text, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
                    tts.save(tmpfile.name)
                    audio_file_path = tmpfile.name
            except Exception as e:
                print("TTS error:", e)
        
        # Draw on frame
        if text:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (600, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"OCR: {text}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.2)


# Face Recognition Functions
def init_face_camera():
    global face_camera
    if face_camera is None:
        face_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return face_camera

def release_face_camera():
    global face_camera
    if face_camera is not None:
        face_camera.release()
        face_camera = None
def generate_face_capture(user_name, num_samples=25):
    global face_capture_active, face_camera
    face_camera = init_face_camera()
    count = 0

    while face_capture_active and count < num_samples:
        success, frame = face_camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]

            # Save image as: User.<username>.<count>.jpg
            file_path = os.path.join(dataset_path, f"User.{user_name}.{count}.jpg")
            cv2.imwrite(file_path, face_img)

            # Draw rectangle and progress text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Capturing {count}/{num_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Break after capturing one face per frame
            break

        # Encode and stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Small delay for stability - FIXED: use time.sleep instead of datetime.time.sleep
        time.sleep(0.2)

    face_capture_active = False
    release_face_camera()
    print(f"[INFO] Face capture complete: {count} images for user {user_name}")


def generate_face_recognition():
    global face_recognition_active, face_camera, recognizer
    
    if recognizer is None:
        yield "data: Face recognition not available. Install opencv-contrib-python.\n\n"
        return
    
    face_camera = init_face_camera()
    
    if not os.path.exists(trainer_path):
        yield "data: No trained model found. Please train first.\n\n"
        return
    
    recognizer.read(trainer_path)
    
    while face_recognition_active:
        success, frame = face_camera.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 60:
               label = id_to_name.get(id_, "Unknown")  # Map ID to name, default to "Unknown"
               color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} - {round(100 - confidence)}%", (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    release_face_camera()

def train_face_model():
    print("[INFO] Training model...")

    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    face_samples, ids = [], []

    for image_path in image_paths:
        try:
            gray_img = Image.open(image_path).convert("L")
            img_np = np.array(gray_img, "uint8")
            # Extract username from filename: User.<name>.<count>.jpg
            user_name = os.path.split(image_path)[-1].split(".")[1]

            faces = face_cascade.detectMultiScale(img_np)
            for (x, y, w, h) in faces:
                face_samples.append(img_np[y:y+h, x:x+w])
                ids.append(user_name)  # Store the string name

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    if face_samples:
        # LBPH recognizer requires numeric IDs, so we map strings to numbers
        unique_names = list(set(ids))
        name_to_id = {name: idx for idx, name in enumerate(unique_names)}
        id_to_name.clear()
        for name, idx in name_to_id.items():
            id_to_name[idx] = name

        numeric_ids = [name_to_id[name] for name in ids]
        recognizer.train(face_samples, np.array(numeric_ids))
        recognizer.save(trainer_path)
        print(f"[INFO] Model trained and saved at {trainer_path}")
        return True
    else:
        print("[ERROR] No faces found for training")
        return False


# Existing Routes
@app.route("/")
def index():
    return render_template("home.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        if not all([name, email, username, password, confirm_password]):
            flash('All fields are required', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (name, email, username, password) VALUES (?, ?, ?, ?)",
                     (name, email, username, hashed_password))
            conn.commit()
            conn.close()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[4], password):
            session['user_id'] = user[0]
            session['username'] = user[3]
            flash('Login successful!', 'success')
            return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/user_dashboard')
def user_dashboard():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    return render_template('user_dashboard.html', username=session['username'])

@app.route('/matching_game')
def matching_game():
    return render_template('matching_game.html')

@app.route('/capture_status')
def capture_status():
    """Check face capture status"""
    user_name = request.args.get('user_name')
    
    # Count how many images have been captured for this user
    count = 0
    if user_name and os.path.exists(dataset_path):
        for filename in os.listdir(dataset_path):
            if filename.startswith(f"User.{user_name}."):
                count += 1
    
    return jsonify({
        'active': face_capture_active,
        'count': count,
        'status': 'complete' if count >= 25 else 'in_progress'
    })

# New OCR Routes
@app.route('/ocr_tool')
def ocr_tool():
    """OCR tool page"""
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    return render_template('ocr_tool.html', username=session['username'])

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    """Initialize and start the camera"""
    global camera, is_camera_active, current_mode
    
    current_mode = "video"
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({'status': 'error', 'message': 'Could not open camera'})
    
    is_camera_active = True
    return jsonify({'status': 'success', 'message': 'Camera started'})

@app.route('/stop_camera')
def stop_camera():
    """Stop the camera"""
    global is_camera_active, camera
    
    is_camera_active = False
    if camera:
        camera.release()
        camera = None
        
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/set_mode/<mode>')
def set_mode(mode):
    """Set the current mode (image or video)"""
    global current_mode
    
    if mode in ['image', 'video']:
        current_mode = mode
        return jsonify({'status': 'success', 'message': f'Mode set to {mode}'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid mode'})

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process uploaded image"""
    global last_text, audio_file_path, current_mode
    
    current_mode = "image"
    
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No image selected'})
    
    try:
        # Read image file
        img = Image.open(file.stream)
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Perform OCR
        text = pytesseract.image_to_string(gray, lang="eng").strip()
        last_text = text
        
        # Convert text to speech
        if text:
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
                tts.save(tmpfile.name)
                audio_file_path = tmpfile.name
            
            # Convert image to base64 for display
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'status': 'success', 
                'text': text,
                'image': f"data:image/jpeg;base64,{img_str}",
                'has_audio': True
            })
        else:
            return jsonify({'status': 'success', 'text': 'No text found', 'has_audio': False})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error processing image: {str(e)}'})

@app.route('/get_audio')
def get_audio():
    """Get the generated audio file"""
    global audio_file_path
    
    if audio_file_path and os.path.exists(audio_file_path):
        return send_file(audio_file_path, as_attachment=True, download_name='speech.mp3')
    else:
        return jsonify({'status': 'error', 'message': 'No audio file available'})

@app.route('/get_text')
def get_text():
    """Get the latest extracted text"""
    global last_text
    return jsonify({'text': last_text})

# ----------------- Admin Routes -----------------
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if email == 'admin@memoryaid.com' and password == 'admin123':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            return redirect(url_for('admin_login'))
    
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        flash('Please login first.', 'warning')
        return redirect(url_for('admin_login'))
    return render_template('admin_dashboard.html')

@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('admin_login'))

# ----------------- Face Recognition Routes -----------------
@app.route('/face_recognition')
def face_recognition():
    """Face recognition page for both admin and regular users"""
    # Check if admin is logged in
    if session.get('admin_logged_in'):
        return render_template('face_recognition.html', username="Admin", user_type="admin")
    
    # Check if regular user is logged in
    if 'user_name' in session:
        return render_template('face_recognition.html', 
                             username=session.get('username', 'User'), 
                             user_type="user")
    
    # If neither is logged in, redirect to login
    flash('Please login first', 'error')
    return redirect(url_for('login'))

@app.route('/face_capture_feed/<user_name>')
def face_capture_feed(user_name):
    return Response(generate_face_capture(user_name), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_face_capture/<user_name>')
def start_face_capture(user_name):
    global face_capture_active
    face_capture_active = True
    return jsonify({'status': 'success', 'message': f'Started capturing faces for user {user_name}'})

@app.route('/stop_face_capture')
def stop_face_capture():
    global face_capture_active
    face_capture_active = False
    release_face_camera()
    return jsonify({'status': 'success', 'message': 'Face capture stopped'})

@app.route('/train_face_model')
def train_face_model_route():
    if recognizer is None:
        return jsonify({'status': 'error', 'message': 'Face recognition not available. Install opencv-contrib-python.'})
    
    success = train_face_model()
    if success:
        return jsonify({'status': 'success', 'message': 'Model trained successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'No faces found for training'})

@app.route('/face_recognition_feed')
def face_recognition_feed():
    return Response(generate_face_recognition(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_face_recognition')
def start_face_recognition():
    global face_recognition_active
    face_recognition_active = True
    return jsonify({'status': 'success', 'message': 'Face recognition started'})

@app.route('/stop_face_recognition')
def stop_face_recognition():
    global face_recognition_active
    face_recognition_active = False
    release_face_camera()
    return jsonify({'status': 'success', 'message': 'Face recognition stopped'})

# Route to serve user.html
@app.route('/user')
def serve_user_html():
    return render_template('user.html')

# Additional training endpoint for the face recognition HTML
@app.route('/train_model', methods=['POST'])
def train_model():
    """API endpoint to train the face recognition model"""
    if recognizer is None:
        return jsonify({'status': 'error', 'message': 'Face recognition not available. Install opencv-contrib-python.'})
    
    # Run training in a separate thread to avoid blocking
    def train_in_background():
        success = train_face_model()
        # You could update a status in database or session here
    
    thread = threading.Thread(target=train_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Training started'})

@app.route('/training_status')
def training_status():
    """Check training status - you'll need to implement actual status tracking"""
    # This is a placeholder - implement actual status tracking
    return jsonify({'status': 'complete', 'message': 'Training completed'})

# ------------------ REMINDER APP ROUTES ------------------ #
# Database setup for reminders
def init_reminder_db():
    conn = sqlite3.connect("reminders.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            time_of_day TEXT,
            details TEXT,
            reminder_time TEXT,
            created_at TEXT,
            sent INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

# Telegram function
def send_to_telegram(message):
    # 🔹 Direct bot credentials (replace with your own if needed)
    BOT_TOKEN = "Replace_with_your_telegram_bot_token"
    CHAT_ID = "Replace_with_your_chat_id"
    # for safety purpose Bot token and chat id are not included in the code. Please replace the placeholders with your actual credentials to enable Telegram functionality.

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ Failed to send message: {e}")
        return False

# Time check function
def is_correct_time(selected_time_str):
    """Check if current time matches the selected reminder time"""
    try:
        # Get current time in the same format as our reminder time
        current_time = datetime.now().strftime("%I:%M %p")
        
        # Compare the times
        return current_time == selected_time_str
    except:
        return False

# Save reminder function
def save_reminder(category, time_of_day, details, reminder_time):
    conn = sqlite3.connect("reminders.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reminders (category, time_of_day, details, reminder_time, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (category, time_of_day, details, reminder_time, datetime.now().strftime('%Y-%m-%d %I:%M %p')))
    conn.commit()
    conn.close()

# Reminder app route - MODIFIED TO NOT REDIRECT TO LOGIN
@app.route('/reminder_app')
def reminder_app():
    # Directly render the template without checking for login
    init_reminder_db()
    
    # Get saved reminders for display
    conn = sqlite3.connect("reminders.db")
    cursor = conn.cursor()
    cursor.execute("SELECT category, time_of_day, details, reminder_time, created_at FROM reminders")
    reminders = cursor.fetchall()
    conn.close()
    
    # Convert to list of dictionaries for easier handling in template
    reminders_list = []
    for reminder in reminders:
        reminders_list.append({
            'category': reminder[0],
            'time_of_day': reminder[1],
            'details': reminder[2],
            'reminder_time': reminder[3],
            'created_at': reminder[4]
        })
    
    return render_template('remind.html', 
                         username="Test User",  # Hardcoded for testing
                         reminders=reminders_list)

# API endpoint to set reminder - MODIFIED TO NOT REQUIRE LOGIN
@app.route('/set_reminder', methods=['POST'])
def set_reminder():
    data = request.get_json()
    category = data.get('category')
    time_of_day = data.get('time_of_day')
    details = data.get('details')
    reminder_time = data.get('reminder_time')  # Get the custom time from frontend
    
    if not all([category, time_of_day, details]):
        return jsonify({'success': False, 'message': 'All fields are required'})
    
    # Handle both preset and custom times
    if time_of_day == 'Custom':
        # For custom times, use the reminder_time directly from frontend
        formatted_time = reminder_time
        try:
            # Validate the custom time format
            datetime.strptime(formatted_time, "%I:%M %p")
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid time format. Use HH:MM AM/PM'})
    else:
        # For preset times, use the fixed times
        fixed_times = {
            "Morning": time(6, 0),    # 6:00 AM
            "Afternoon": time(15, 0), # 3:00 PM
            "Evening": time(20, 0)    # 8:00 PM
        }
        
        if time_of_day not in fixed_times:
            return jsonify({'success': False, 'message': 'Invalid time selection'})
        
        fixed_time = fixed_times[time_of_day]
        formatted_time = fixed_time.strftime("%I:%M %p")
    
    # Check if current time matches the selected reminder time
    current_time_str = datetime.now().strftime("%I:%M %p")
    
    if not is_correct_time(formatted_time):
        return jsonify({
            'success': False, 
            'message': f"❌ Reminder not sent! Current time is {current_time_str}. Reminders can only be sent at exactly {formatted_time}."
        })
    
    # Save to DB
    save_reminder(category, time_of_day, details, formatted_time)

    # Create reminder text
    reminder_text = (
        f"⏰ Reminder!\n"
        f"Category: {category}\n"
        f"Time of Day: {time_of_day}\n"
        f"Task: {details}\n"
        f"Scheduled Time: {formatted_time}\n\n"
        f"Sent At: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
    )

    # Send to Telegram
    if send_to_telegram(reminder_text):
        return jsonify({'success': True, 'message': 'Reminder sent successfully! ✅'})
    else:
        return jsonify({'success': False, 'message': 'Failed to send reminder to Telegram'})

# ------------------ GPS TRACKING AND GEOFENCING ROUTES ------------------ #

@app.route('/index1')
def index1():
    return render_template('home.html')

@app.route('/index2')
def index2():
    return render_template('map.html')

@app.route('/index3')
def index3():
    return render_template('gps_dashboard.html')

def get_geofence_polygon():
    """Fetch polygon coordinates from DB and return as Shapely Polygon."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT lat, lng FROM geofence")
    coords = cursor.fetchall()
    conn.close()

    if not coords:
        return None

    # Shapely polygon expects (lon, lat) format
    coords_lon_lat = [(lng, lat) for lat, lng in coords]
    return Polygon(coords_lon_lat)

def send_telegram_alert(lat, lon):
    """Send alert to parent via Telegram with Google Maps link."""
    gmap_link = f"https://maps.google.com/?q={lat},{lon}"
    message = (
        f"⚠️ ALERT: Patient is outside safe zone!\n"
        f"Location: {lat}, {lon}\n"
        f"Map: {gmap_link}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.get(url, params=params, timeout=2)
    except Exception as e:
        print("Error sending Telegram alert:", e)

@app.route("/data")
def get_gps_data():
    lat = lon = None
    try:
        response = requests.get(URL, timeout=3)
        if response.status_code == 200:
            gps_data = response.json()
            print("Received GPS data:", gps_data)

            lat = float(gps_data.get("lat"))
            lon = float(gps_data.get("lon"))

            polygon = get_geofence_polygon()
            if polygon:
                point = Point(lon, lat)
                if not polygon.contains(point):
                    send_telegram_alert(lat, lon)
                    return jsonify({"status": "alert", "lat": lat, "lon": lon})

            return jsonify({"status": "safe", "lat": lat, "lon": lon})

        else:
            return jsonify({"status": "error", "lat": lat, "lon": lon, "message": f"Device error {response.status_code}"})

    except Exception as e:
        return jsonify({"status": "error", "lat": lat, "lon": lon, "message": str(e)})

@app.route('/save-fence', methods=['POST'])
def save_fence():
    data = request.get_json()
    coordinates = data.get('coordinates', [])

    with sqlite3.connect('geofence.db') as conn:
        conn.execute('DELETE FROM geofence')  # Clear previous geofence
        conn.executemany('INSERT INTO geofence (lat, lng) VALUES (?, ?)', coordinates)

    return jsonify({'status': 'Geofence saved successfully'})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)