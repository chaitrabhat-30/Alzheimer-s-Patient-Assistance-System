import streamlit as st
import requests
import sqlite3
import pandas as pd
from datetime import datetime, time

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="Reminder App",
    page_icon="⏰",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------ #
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        height: 3rem;
        width: 100%;
        border-radius: 10px;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        color: #155724;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .time-display {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #e6f3ff;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        padding: 1rem;
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------ #
st.markdown('<h1 class="main-header">⏰ Reminder App</h1>', unsafe_allow_html=True)

# ------------------ DB SETUP ------------------ #
def init_db():
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

def save_reminder(category, time_of_day, details, reminder_time):
    conn = sqlite3.connect("reminders.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reminders (category, time_of_day, details, reminder_time, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (category, time_of_day, details, reminder_time, datetime.now().strftime('%Y-%m-%d %I:%M %p')))
    conn.commit()
    conn.close()

# ------------------ TELEGRAM ------------------ #
def send_to_telegram(message):
    # 🔹 Direct bot credentials (replace with your own if needed)
    BOT_TOKEN = "replace_with_your_bot_token"
    # removed for security reasons, you should replace it with your own chat ID
    CHAT_ID = "replace_with_your_chat_id"

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"❌ Failed to send message: {e}")
        return False

# ------------------ TIME CHECK FUNCTION ------------------ #
def is_correct_time(selected_time_str):
    """Check if current time matches the selected reminder time"""
    try:
        # Get current time in the same format as our reminder time
        current_time = datetime.now().strftime("%I:%M %p")
        
        # Compare the times
        return current_time == selected_time_str
    except:
        return False

# ------------------ APP LOGIC ------------------ #
init_db()

st.subheader("Choose Category:")

col1, col2 = st.columns(2)

# Track category choice
if "category" not in st.session_state:
    st.session_state.category = None

with col1:
    if st.button("🏃 Physical Activity"):
        st.session_state.category = "Physical Activity"

with col2:
    if st.button("💊 Medicine"):
        st.session_state.category = "Medicine"

# If category selected, show next options
if st.session_state.category:
    st.info(f"Selected Category: **{st.session_state.category}**")

    # Time of Day
    time_of_day = st.radio("Select Time of Day:", ["Morning", "Afternoon", "Evening"], horizontal=True)
    
    # Set fixed times for each time of day
    fixed_times = {
        "Morning": time(6, 0),    # 6:00 AM
        "Afternoon": time(15, 0), # 3:00 PM
        "Evening": time(20, 0)    # 8:00 PM
    }
    
    # Display the fixed time
    fixed_time = fixed_times[time_of_day]
    formatted_time = fixed_time.strftime("%I:%M %p")
    st.markdown(f'<div class="time-display">⏰ Reminder Time: {formatted_time}</div>', unsafe_allow_html=True)
    
    # Store the formatted time in session state
    if "reminder_time" not in st.session_state:
        st.session_state.reminder_time = formatted_time
    else:
        st.session_state.reminder_time = formatted_time

    # Activity / Medicine input
    details = st.text_input(f"Enter {st.session_state.category} details:")

    # Submit button
    if st.button("✅ Set Reminder"):
        if not details:
            st.warning("⚠️ Please enter the details.")
        else:
            # Check if current time matches the selected reminder time
            if not is_correct_time(st.session_state.reminder_time):
                st.markdown(f'<div class="error-message">❌ Reminder not sent! Current time is {datetime.now().strftime("%I:%M %p")}. Reminders can only be sent at exactly {st.session_state.reminder_time}.</div>', unsafe_allow_html=True)
            else:
                # Save to DB
                save_reminder(st.session_state.category, time_of_day, details, st.session_state.reminder_time)

                # Create reminder text
                reminder_text = (
                    f"⏰ Reminder!\n"
                    f"Category: {st.session_state.category}\n"
                    f"Time of Day: {time_of_day}\n"
                    f"Task: {details}\n\n"
                    f"Sent At: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
                )

                # Send to Telegram
                if send_to_telegram(reminder_text):
                    st.markdown('<div class="success-message">Reminder sent successfully! ✅</div>', unsafe_allow_html=True)

# ------------------ SHOW SAVED REMINDERS ------------------ #
if st.checkbox("📋 Show Saved Reminders"):
    conn = sqlite3.connect("reminders.db")
    reminders = conn.execute("SELECT category, time_of_day, details, reminder_time, created_at FROM reminders").fetchall()
    conn.close()

    if reminders:
        df = pd.DataFrame(reminders, columns=["Category", "Time of Day", "Details", "Reminder Time", "Created At"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No reminders saved yet.")