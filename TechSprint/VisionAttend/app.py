import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time 
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from scipy.spatial import distance as dist
import google.generativeai as genai
import random

# Page Config
st.set_page_config(page_title="VisionAttend (Firebase)", layout="wide")

import streamlit.components.v1 as components

# Custom CSS/JS for Cursor Trail
def add_cursor_trail():
    js = """
    <script>
    (function() {
        var parentDoc = window.parent.document;
        var old = parentDoc.getElementById('trail-canvas');
        if (old) old.remove();
        
        var canvas = parentDoc.createElement('canvas');
        canvas.id = 'trail-canvas';
        Object.assign(canvas.style, {
            position: 'fixed', top: '0', left: '0',
            width: '100vw', height: '100vh',
            pointerEvents: 'none', zIndex: '9999999'
        });
        parentDoc.body.appendChild(canvas);
        
        var ctx = canvas.getContext('2d');
        var width, height;
        
        function resize() {
            width = canvas.width = window.parent.innerWidth;
            height = canvas.height = window.parent.innerHeight;
        }
        window.parent.addEventListener('resize', resize);
        resize();
        
        var particles = [];
        var mouse = { x: 0, y: 0 };
        var lastMouse = { x: 0, y: 0 };
        
        parentDoc.addEventListener('mousemove', function(e) {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        });
        
        function animate() {
            ctx.clearRect(0, 0, width, height);
            
            // Calculate distance moved
            var dist = Math.hypot(mouse.x - lastMouse.x, mouse.y - lastMouse.y);
            
            // Spawn particles based on movement
            if (dist > 2) {
                for(let i=0; i < Math.min(dist/2, 5); i++) {
                    particles.push({
                        x: mouse.x + (Math.random()-0.5)*10,
                        y: mouse.y + (Math.random()-0.5)*10,
                        vx: (Math.random() - 0.5) * 1.5,
                        vy: (Math.random() - 0.5) * 1.5 + 0.5, // Slight gravity
                        life: 1.0,
                        size: Math.random() * 3 + 1,
                        hue: Math.random() * 60 + 30 // Gold/Yellow/Orange spectrum
                    });
                }
            }
            lastMouse.x = mouse.x;
            lastMouse.y = mouse.y;
            
            for(var i=0; i<particles.length; i++) {
                var p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.life -= 0.015;
                p.size -= 0.03;
                
                if(p.life <= 0 || p.size <= 0) {
                    particles.splice(i, 1);
                    i--;
                    continue;
                }
                
                ctx.fillStyle = `hsla(${p.hue}, 100%, 60%, ${p.life})`;
                ctx.shadowBlur = 10;
                ctx.shadowColor = `hsla(${p.hue}, 100%, 50%, 1)`;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI*2);
                ctx.fill();
                ctx.shadowBlur = 0; // Reset
            }
            requestAnimationFrame(animate);
        }
        animate();
    })();
    </script>
    """
    components.html(js, height=0, width=0)

# Call the enhancement
add_cursor_trail()

# Title and Sidebar
st.title("VisionAttend: Intelligent Attendance System (Firebase)")
st.sidebar.header("Controls")
run_camera = st.sidebar.checkbox("Start Camera", value=False)
show_encodings = st.sidebar.checkbox("Show Known Faces", value=False)
enable_liveness = st.sidebar.checkbox("Enable Liveness Detection (Blink)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Google Tech Integration 🍌")
# API Key handling: Check environment variable or use offline mode automatically
google_api_key = os.getenv("GOOGLE_API_KEY") 
enable_smart_greetings = st.sidebar.checkbox("Enable Smart Greetings", value=True)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, 'images')
SERVICE_ACCOUNT_KEY_PATH = os.path.join(BASE_DIR, 'serviceAccountKey.json')

# Firebase Setup
# Use caching to prevent re-initializing app on every run
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
            if os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
                cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
                firebase_admin.initialize_app(cred)
                return firestore.client()
            else:
                st.error("❌ `serviceAccountKey.json` not found! Please place it in the project root.")
                return None
        return firestore.client()
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        return None

db = init_firebase()

# Helper Functions
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def generate_smart_greeting(name):
    if not google_api_key:
        # Fallback for offline mode
        greetings = [
            f"Welcome back, {name}! (Offline Mode)",
            f"Hello {name}, ready to code?",
            f"Greetings {name}! The system recognizes you.",
            f"Hi {name}, looking sharp today!"
        ]
        return random.choice(greetings)
    
    try:
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        now = datetime.now()
        time_of_day = "morning" if 5 <= now.hour < 12 else "afternoon" if 12 <= now.hour < 18 else "evening"
        
        prompt = f"Write a short, witty, and motivating sentence to welcome '{name}' to class/work. It is currently {time_of_day}. Keep it under 20 words."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Welcome, {name}! (Gemini Error: {str(e)})"

def generate_nano_banana_fact():
    if not google_api_key:
        # Fallback for offline mode
        facts = [
            "Nano Banana says: Bananas share 50% of their DNA with humans. So do bad coders.",
            "Nano Banana Wisdom: A bug in the code is worth two in the documentation.",
            "Nano Banana Fact: The first computer bug was an actual moth.",
            "Nano Banana Joke: Why did the programmer quit his job? because he didn't get arrays.",
            "Nano Banana Tip: Always comment your code, unlike this banana."
        ]
        return random.choice(facts)
    
    try:
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = "Tell me a short, funny, banana-themed computer science fact or joke. Call it 'Nano Banana Wisdom'."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Nano Banana is sleeping. (Error: {str(e)})"

@st.cache_data
def load_encodings(path):
    images = []
    classNames = []
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    myList = os.listdir(path)
    encodeList = []
    
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is not None:
            curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
            try:
                encode = face_recognition.face_encodings(curImg)[0]
                encodeList.append(encode)
                classNames.append(os.path.splitext(cl)[0])
            except IndexError:
                continue
                
    return encodeList, classNames

def mark_attendance_firebase(name):
    if db is None:
        return False, "Database not connected"
        
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    
    # Doc ID: Name_Date to ensure uniqueness for the day
    doc_id = f"{name}_{date_str}"
    
    doc_ref = db.collection('attendance').document(doc_id)
    doc = doc_ref.get()
    
    if doc.exists:
        return False, "Already marked for today"
    else:
        # Create new record
        doc_ref.set({
            'name': name,
            'time': time_str,
            'date': date_str,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        return True, None

# Load Data
encodeListKnown, classNames = load_encodings(IMAGES_PATH)
st.sidebar.success(f"Loaded {len(classNames)} known faces.")

if show_encodings:
    st.sidebar.write(classNames)
    
if st.sidebar.button("🍌 Nano Banana Mode"):
    with st.spinner("Consulting the Banana Oracle..."):
        fact = generate_nano_banana_fact()
        st.balloons()
        st.sidebar.markdown(f"**Nano Banana says:**\n\n>{fact}")

# UI Layout
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Attendance Log (Firestore)")
    attendance_placeholder = st.empty()

    def update_attendance_view():
        if db is None:
            attendance_placeholder.error("Firebase not connected.")
            return

        # Fetch last 10 records
        # Use 'timestamp' for sorting to avoid needing a composite index in Firestore
        docs = db.collection('attendance').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).stream()
        
        data = []
        for doc in docs:
            data.append(doc.to_dict())
            
        if data:
            df = pd.DataFrame(data)
            # Reorder columns if needed
            if 'timestamp' in df.columns:
                df = df.drop(columns=['timestamp'])
            attendance_placeholder.dataframe(df, use_container_width=True)
        else:
            attendance_placeholder.info("No attendance records found.")

    update_attendance_view()

# State
if 'last_greeting_time' not in st.session_state:
    st.session_state.last_greeting_time = 0
if 'last_greeting_name' not in st.session_state:
    st.session_state.last_greeting_name = ""

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2

with col1:
    st.subheader("Live Feed")
    FRAME_WINDOW = st.image([])
    status_text = st.empty()
    
    if run_camera:
        cap = cv2.VideoCapture(0)
        
        # Blink vars
        COUNTER = 0
        TOTAL_BLINKS = 0
        
        while run_camera:
            success, img = cap.read()
            if not success:
                st.error("Camera not accessible!")
                break
                
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
            # 1. Face Detection & Recognition
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            
            # 2. Landmarks for Liveness
            face_landmarks_list = face_recognition.face_landmarks(imgS, facesCurFrame)
            
            detected_names = []
            
            for (encodeFace, faceLoc), landmarks in zip(zip(encodesCurFrame, facesCurFrame), face_landmarks_list):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                
                name = "Unknown"
                color = (255, 0, 0) # Red
                
                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        color = (0, 255, 0) # Green (Potential)

                        # LIVENESS CHECK
                        is_live = not enable_liveness # If liveness disabled, assume live
                        
                        if enable_liveness:
                            leftEye = landmarks['left_eye']
                            rightEye = landmarks['right_eye']
                            
                            ear_left = eye_aspect_ratio(leftEye)
                            ear_right = eye_aspect_ratio(rightEye)
                            ear = (ear_left + ear_right) / 2.0
                            
                            if ear < EYE_AR_THRESH:
                                COUNTER += 1
                            else:
                                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                    TOTAL_BLINKS += 1
                                    st.toast("Blink Detected! Live.")
                                COUNTER = 0
                                
                            # Display Blink Count
                            cv2.putText(img, f"Blinks: {TOTAL_BLINKS}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            if TOTAL_BLINKS >= 1:
                                is_live = True
                                color = (0, 255, 0)
                            else:
                                color = (255, 165, 0) # Orange: Waiting for blink
                                cv2.putText(img, "BLINK TO VERIFY", (faceLoc[3]*4, faceLoc[0]*4 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                        detected_names.append(name)

                        if is_live:
                            marked_new, msg = mark_attendance_firebase(name)
                            if marked_new:
                                update_attendance_view()
                                if enable_smart_greetings and google_api_key:
                                    greeting = generate_smart_greeting(name)
                                    st.toast(greeting, icon="🍌")
                                else:
                                    st.toast(f"Welcome {name}!")
                            elif msg: # If not new, but there's a message (e.g., already marked)
                                st.toast(msg)
                            
                # Draw Rectangle
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Reset blinks if face lost?
            if not detected_names:
                pass 
                
        cap.release()
    else:
        st.info("Check 'Start Camera' to begin.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Note**: Liveness detection & Nano Banana included.")

# About Section
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.markdown("""
**Contact Details:**
- **Mobile No:** 8669223528
- **Email ID:** aadiagarwal911@gmail.com
- **Origin:** India 🇮🇳
""")
