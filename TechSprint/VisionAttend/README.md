# VisionAttend System

This is a production-ready Face Attendance System with Liveness Detection, now powered by **Google Firebase** for real-time cloud database storage.

## Setup

1. **Add Images**:
   - Place photos of known people in the `images` folder.
   - The filename (e.g., `Elon Musk.jpg`) will be used as the person's name.

2. **Firebase Configuration**:
   - Ensure `serviceAccountKey.json` is present in the project root. (Get this from Firebase Console > Project Settings > Service Accounts).

3. **Run the Application**:
   - Open a terminal in this directory.
   - Run: `cd VisionAttend`
   - Run: `streamlit run app.py`

## Features

- **Face Recognition**: Identifies known faces from the `images` folder.
- **Cloud Attendance Logging**: Automatically logs Name, Time, and Date to **Firebase Firestore**.
- **Liveness Detection**: Prevents spoofing by checking for eye blinks.
   - **Instructions**: When "Unknown" or "Orange" box appears, blink your eyes naturally to verify liveness.
   - Once verified, the box turns Green and attendance is marked.
- **Dashboard**: View the live camera feed and the latest attendance log from Firestore in real-time.

## Alternative (Core Script)
If you prefer the command-line version without the GUI:
- Run: `python main.py`
