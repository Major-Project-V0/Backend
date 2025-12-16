import os
import cv2
import csv
import numpy as np
import time
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions.drawing_utils import draw_landmarks

# --- Configuration ---
labels = ["appropriate", "cheating", "defensive"]   # The 3 classes
samples_per_class = 2000                            # 2000 rows per class
output_csv = "landmarks.csv"

# Create dataset_images folder if not exists
os.makedirs("dataset_images", exist_ok=True)

# --- Helper function for flattening ---
def flatten_landmarks(landmarks, target_count, use_visibility=False):
    """
    Flattens Mediapipe landmarks into (x,y,z,visibility).
    Face & hands → visibility=1.0, Pose → real visibility.
    """
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


# --- Collect Data for Each Label ---
all_data_rows = []

cap = cv2.VideoCapture(0)

with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for label in labels:
        print(f"\n[▶] Starting collection for label: {label}")
        image_dir = os.path.join("dataset_images", label)
        os.makedirs(image_dir, exist_ok=True)

        last_saved_time = time.time()
        image_count = 0
        collected = 0

        while cap.isOpened() and collected < samples_per_class:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            # Only save if landmarks detected
            if results.pose_landmarks and results.face_landmarks:
                current_time = time.time()
                if current_time - last_saved_time >= 0.4:  # every 0.2 seconds
                    face = flatten_landmarks(results.face_landmarks, 468, use_visibility=False)
                    pose = flatten_landmarks(results.pose_landmarks, 33, use_visibility=True)
                    left = flatten_landmarks(results.left_hand_landmarks, 21, use_visibility=False)
                    right = flatten_landmarks(results.right_hand_landmarks, 21, use_visibility=False)

                    row = np.concatenate([face, pose, left, right])
                    row = np.append(row, label)
                    all_data_rows.append(row)

                    # Save image
                    image_path = os.path.join(image_dir, f"{label}_{image_count}.jpg")
                    cv2.imwrite(image_path, frame)
                    print(f"[✓] Saved {image_path} ({collected+1}/{samples_per_class})")
                    image_count += 1
                    collected += 1

                    last_saved_time = current_time

            # Show window
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print(f"[✅] Completed {samples_per_class} samples for {label}")

cap.release()
cv2.destroyAllWindows()

# --- Save Final CSV (all classes together) ---
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_data_rows)

print(f"\n[🎯] Dataset collection complete. Total rows = {len(all_data_rows)} saved in '{output_csv}'")
