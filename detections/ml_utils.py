"""
ML Utilities for running posture and face detection models
"""
import os
import sys
import cv2
import numpy as np
import joblib
import mediapipe as mp
from pathlib import Path

# Add Models directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "Models"

# Initialize MediaPipe components
mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Global variables for loaded models (singleton pattern)
_posture_model = None
_posture_encoder = None
_holistic = None
_face_detection = None


def load_posture_model():
    """Load the posture detection model and encoder (singleton)"""
    global _posture_model, _posture_encoder
    
    if _posture_model is None or _posture_encoder is None:
        model_path = MODELS_DIR / "ridge_model.pkl"
        encoder_path = MODELS_DIR / "label_encoder.pkl"
        
        if not model_path.exists() or not encoder_path.exists():
            raise FileNotFoundError(f"Model files not found in {MODELS_DIR}")
        
        _posture_model = joblib.load(model_path)
        _posture_encoder = joblib.load(encoder_path)
        print("Posture model loaded successfully")
    
    return _posture_model, _posture_encoder


def get_holistic_processor():
    """Get or create MediaPipe Holistic processor (singleton)"""
    global _holistic
    
    if _holistic is None:
        _holistic = mp_holistic.Holistic(
            static_image_mode=True,  # Use True for single image processing
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    return _holistic


def get_face_detection_processor():
    """Get or create MediaPipe Face Detection processor (singleton)"""
    global _face_detection
    
    if _face_detection is None:
        _face_detection = mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
    
    return _face_detection


def flatten_landmarks(landmarks, target_count, use_visibility=False):
    """Flatten MediaPipe landmarks into (x,y,z,visibility) array"""
    data = []
    if landmarks:
        for lm in landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
            data.append(lm.visibility if use_visibility and hasattr(lm, 'visibility') else 1.0)
        while len(data) < target_count * 4:
            data.extend([0, 0, 0, 1.0])
    else:
        data = [0] * (target_count * 4)
    return np.array(data[:target_count * 4])


def extract_posture_features(results):
    """Extract features from MediaPipe Holistic results"""
    # Face first
    face = flatten_landmarks(results.face_landmarks, 468, use_visibility=False)
    # Pose second
    pose = flatten_landmarks(results.pose_landmarks, 33, use_visibility=True)
    # Left hand
    left = flatten_landmarks(results.left_hand_landmarks, 21, use_visibility=False)
    # Right hand
    right = flatten_landmarks(results.right_hand_landmarks, 21, use_visibility=False)
    
    # Concatenate in the same order as dataset
    return np.concatenate([face, pose, left, right]).reshape(1, -1)


def detect_posture(image_array):
    """
    Detect posture from image array (numpy array in RGB format)
    
    Args:
        image_array: numpy array of image in RGB format (from cv2.imread or base64 decode)
    
    Returns:
        dict with 'posture_label' and 'confidence' (or None if no landmarks detected)
    """
    try:
        # Load models
        clf, le = load_posture_model()
        
        # Get holistic processor
        holistic = get_holistic_processor()
        
        # Ensure image is in RGB format
        if len(image_array.shape) == 3:
            # If BGR, convert to RGB
            if image_array.dtype == np.uint8:
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image_array
        else:
            raise ValueError("Image must be a 3-channel RGB image")
        
        # Process image
        results = holistic.process(rgb_image)
        
        # Extract features
        landmarks = extract_posture_features(results)
        
        # Only predict if we have valid landmarks
        if landmarks.sum() == 0:
            return None
        
        # Predict
        y_pred = clf.predict(landmarks)
        pred_class = le.inverse_transform(y_pred)[0]
        
        # Get confidence if model supports it
        confidence = None
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(landmarks)[0]
            confidence = float(np.max(proba))
        
        return {
            'posture_label': pred_class,
            'confidence': confidence
        }
    
    except Exception as e:
        print(f"Error in detect_posture: {str(e)}")
        raise


def detect_faces(image_array):
    """
    Detect faces from image array (numpy array in RGB format)
    
    Args:
        image_array: numpy array of image in RGB format
    
    Returns:
        dict with 'face_count' and 'detections' list
    """
    try:
        # Get face detection processor
        face_detection = get_face_detection_processor()
        
        # Ensure image is in RGB format
        if len(image_array.shape) == 3:
            if image_array.dtype == np.uint8:
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image_array
        else:
            raise ValueError("Image must be a 3-channel RGB image")
        
        # Process image
        results = face_detection.process(rgb_image)
        
        # Extract detection data
        detections_list = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                detections_list.append({
                    'confidence': float(detection.score[0]),
                    'bbox': {
                        'x': float(bbox.xmin),
                        'y': float(bbox.ymin),
                        'width': float(bbox.width),
                        'height': float(bbox.height)
                    }
                })
        
        return {
            'face_count': len(detections_list),
            'detections': detections_list
        }
    
    except Exception as e:
        print(f"Error in detect_faces: {str(e)}")
        raise


def base64_to_image(base64_string):
    """
    Convert base64 encoded image string to numpy array
    
    Args:
        base64_string: base64 encoded image (with or without data URL prefix)
    
    Returns:
        numpy array of image in BGR format (OpenCV format)
    """
    import base64
    
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image from base64 string")
    
    return image

