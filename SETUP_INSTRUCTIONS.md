# Setup Instructions for ML Detection Integration

## Overview
This integration connects the ML models (live_posture.py and multi_face_detection.py) with the Django backend and React frontend. The models run on the backend, processing images sent from the frontend, and results are stored in the database.

## Backend Setup

### 1. Install Required Packages (if not already installed)
Make sure you have all required packages in your virtual environment:

```bash
cd /Users/ranjit/Desktop/Collegeproject/Backend
source env/bin/activate  # or .venv/bin/activate depending on your setup

pip install django djangorestframework django-cors-headers python-decouple
pip install opencv-python mediapipe==0.10.7 joblib scikit-learn numpy
```

### 2. Run Database Migrations

```bash
python manage.py makemigrations detections
python manage.py migrate
```

### 3. Create Superuser (optional, for admin access)

```bash
python manage.py createsuperuser
```

### 4. Start Django Server

```bash
python manage.py runserver
```

The backend will run on `http://localhost:8000`

## Frontend Setup

### 1. Install Dependencies (if not already installed)

```bash
cd /Users/ranjit/Desktop/Collegeproject/Frontend
npm install
```

### 2. Start Frontend Development Server

```bash
npm run dev
```

The frontend will run on `http://localhost:5173` (or the port shown in terminal)

## How It Works

### API Endpoints

1. **POST /api/detections/both/**
   - Detects both posture and faces from an image
   - Request body: `{ "image": "base64_encoded_image", "session_id": "optional_session_id" }`
   - Returns: Posture label, confidence, face count, and detections

2. **POST /api/detections/posture/**
   - Detects only posture
   - Same request format as above

3. **POST /api/detections/faces/**
   - Detects only faces
   - Same request format as above

4. **GET /api/detections/session/<session_id>/**
   - Retrieves all detections for a session

### Database Models

- **PostureDetection**: Stores posture detection results (appropriate/cheating/defensive)
- **FaceDetection**: Stores face detection results (face count and detection details)
- **DetectionSession**: Tracks interview/detection sessions

### Frontend Integration

The `interview.jsx` component:
- Captures frames from webcam every 1 second when recording
- Converts frames to base64 format
- Sends frames to `/api/detections/both/` endpoint
- Displays results (posture label and face count) as overlay on video
- Stores all detections in database with session_id

## Testing

1. Start both backend and frontend servers
2. Navigate to the interview page in your browser
3. Start camera and recording
4. You should see:
   - Posture detection results (appropriate/cheating/defensive)
   - Face count
   - Results stored in database

## Troubleshooting

### ModuleNotFoundError
- Make sure virtual environment is activated
- Install missing packages: `pip install <package_name>`

### Camera not working
- Check browser permissions for camera access
- Make sure HTTPS is used (required for getUserMedia in some browsers)

### API connection errors
- Verify backend is running on port 8000
- Check CORS settings in Django settings.py
- Check browser console for errors

### Model files not found
- Ensure `ridge_model.pkl` and `label_encoder.pkl` are in `Backend/Models/` directory

### MediaPipe parsing errors
- If you see "Failed to parse: node {...}" error, see `MEDIAPIPE_TROUBLESHOOTING.md` for detailed solutions
- Quick fix: `pip uninstall mediapipe -y && pip install mediapipe==0.10.7`
- Or try: `pip install --upgrade mediapipe` to use the latest version

