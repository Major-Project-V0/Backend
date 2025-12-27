# MediaPipe Troubleshooting Guide

## Error: "Failed to parse: node {...}"

This error occurs when MediaPipe fails to parse its internal graph configuration. This is typically caused by:

1. **Incompatible MediaPipe version**
2. **Missing or corrupted model files**
3. **System compatibility issues**

## Solutions

### Solution 1: Reinstall MediaPipe (Recommended)

```bash
cd /Users/ranjit/Desktop/Collegeproject/Backend
source env/bin/activate  # or .venv/bin/activate

# Uninstall current version
pip uninstall mediapipe -y

# Clear pip cache
pip cache purge

# Reinstall MediaPipe
pip install mediapipe==0.10.7

# If that doesn't work, try the latest version
pip install --upgrade mediapipe
```

### Solution 2: Update to Latest MediaPipe Version

If version 0.10.7 is causing issues, try the latest version:

```bash
pip install --upgrade mediapipe
```

### Solution 3: Install System Dependencies (macOS)

On macOS, you may need additional dependencies:

```bash
# Install via Homebrew
brew install protobuf

# Or via pip
pip install protobuf
```

### Solution 4: Check Python Version

MediaPipe requires Python 3.8-3.11. Check your version:

```bash
python --version
```

If you're using Python 3.12+, you may need to downgrade or use a different MediaPipe version.

### Solution 5: Verify Installation

Test MediaPipe installation:

```python
import mediapipe as mp
print(mp.__version__)

# Test Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True)
print("MediaPipe Holistic initialized successfully!")
holistic.close()
```

### Solution 6: Use Alternative Initialization

If the error persists, the code has been updated to try alternative initialization methods with different model complexities. The error handling will now:

1. Try with default settings (model_complexity=1)
2. Fall back to minimal settings (model_complexity=0) if that fails
3. Provide detailed error messages

### Solution 7: Check for Conflicting Packages

Sometimes other packages can conflict with MediaPipe:

```bash
# Check for conflicts
pip check

# If conflicts found, try creating a fresh virtual environment
python -m venv env_new
source env_new/bin/activate
pip install django djangorestframework django-cors-headers python-decouple
pip install opencv-python mediapipe joblib scikit-learn numpy
```

## Common Error Messages and Fixes

### "Failed to parse: node {...}"
- **Fix**: Reinstall MediaPipe (Solution 1 or 2)

### "No module named 'mediapipe'"
- **Fix**: `pip install mediapipe`

### "AttributeError: module 'mediapipe' has no attribute 'solutions'"
- **Fix**: Reinstall MediaPipe - installation may be corrupted

### "RuntimeError: MediaPipe Holistic initialization failed"
- **Fix**: Check Python version, reinstall MediaPipe, check system dependencies

## Still Having Issues?

1. Check the full error traceback in your Django server logs
2. Verify all dependencies are installed: `pip list | grep mediapipe`
3. Try running a simple MediaPipe test script outside Django
4. Check MediaPipe GitHub issues: https://github.com/google/mediapipe/issues

## Test Script

Create a file `test_mediapipe.py`:

```python
import mediapipe as mp
import cv2
import numpy as np

print(f"MediaPipe version: {mp.__version__}")

# Test Holistic
mp_holistic = mp.solutions.holistic
try:
    holistic = mp_holistic.Holistic(static_image_mode=True)
    print("✓ Holistic initialized successfully")
    
    # Test with a dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = holistic.process(dummy_image)
    print("✓ Holistic processing works")
    
    holistic.close()
except Exception as e:
    print(f"✗ Holistic failed: {e}")

# Test Face Detection
mp_face = mp.solutions.face_detection
try:
    face_detection = mp_face.FaceDetection()
    print("✓ Face Detection initialized successfully")
    
    results = face_detection.process(dummy_image)
    print("✓ Face Detection processing works")
    
    face_detection.close()
except Exception as e:
    print(f"✗ Face Detection failed: {e}")

print("Test complete!")
```

Run it:
```bash
python test_mediapipe.py
```

