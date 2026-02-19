"""
STEP 1 — Extract Keypoints for Shoulder Abduction
===================================================
Processes your dataset images and saves a CSV of keypoints + angles.

Folder structure:
    dataset/
        correct/    ← arm raised correctly
        incorrect/  ← arm not raised / wrong position

Output: keypoints_dataset.csv

Install:
    pip install mediapipe opencv-python pandas numpy
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH  = "/Users/lait-zet/Desktop/pose-correction-ai/mediapip3/pose_landmarker.task"   # ← your .task file
DATASET_DIR = "/Users/lait-zet/Desktop/pose-correction-ai/mediapip3/dataset"
OUTPUT_CSV  = "keypoints_dataset.csv"

CLASSES = {"correct": 1, "incorrect": 0}

# ── Landmark indices for shoulder abduction ───────────────────────
# We only use upper body landmarks (11–16) + a few torso points
UPPER_BODY_IDS = [
    11, 12,   # shoulders
    13, 14,   # elbows
    15, 16,   # wrists
    23, 24,   # hips (for torso reference)
]

# ── MediaPipe setup ───────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# ── Angle calculation ─────────────────────────────────────────────
def calc_angle(a, b, c):
    """
    Calculates angle at point B, formed by A-B-C.
    a, b, c are (x, y) tuples.
    Returns angle in degrees.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_features(image_path):
    """
    Returns feature vector:
    - x,y,z for each upper body landmark (8 landmarks × 3 = 24 values)
    - left arm angle at elbow
    - right arm angle at elbow
    - left shoulder abduction angle (shoulder-hip horizontal)
    - right shoulder abduction angle
    Total: 24 + 4 = 28 features
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    lms = result.pose_landmarks[0]

    # ── Raw keypoints (upper body only) ──────────────────────────
    keypoints = []
    for idx in UPPER_BODY_IDS:
        if idx < len(lms):
            keypoints.extend([lms[idx].x, lms[idx].y, lms[idx].z])
        else:
            keypoints.extend([0.0, 0.0, 0.0])  # fallback

    # ── Computed angles ───────────────────────────────────────────
    def get_xy(idx):
        return (lms[idx].x, lms[idx].y) if idx < len(lms) else (0.0, 0.0)

    l_shoulder = get_xy(11)
    r_shoulder = get_xy(12)
    l_elbow    = get_xy(13)
    r_elbow    = get_xy(14)
    l_wrist    = get_xy(15)
    r_wrist    = get_xy(16)
    l_hip      = get_xy(23)
    r_hip      = get_xy(24)

    # Elbow angles (shoulder → elbow → wrist)
    left_elbow_angle  = calc_angle(l_shoulder, l_elbow, l_wrist)
    right_elbow_angle = calc_angle(r_shoulder, r_elbow, r_wrist)

    # Shoulder abduction angle (hip → shoulder → elbow)
    # This captures how high the arm is raised relative to the torso
    left_abduction  = calc_angle(l_hip, l_shoulder, l_elbow)
    right_abduction = calc_angle(r_hip, r_shoulder, r_elbow)

    angles = [left_elbow_angle, right_elbow_angle,
              left_abduction,   right_abduction]

    return keypoints + angles  # 28 features total

# ── Process dataset ───────────────────────────────────────────────
rows = []
skipped = 0

for label_name, label_value in CLASSES.items():
    folder = os.path.join(DATASET_DIR, label_name)
    if not os.path.exists(folder):
        print(f"⚠️  Folder not found: {folder}")
        continue

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Processing '{label_name}': {len(files)} images...")

    for filename in files:
        path = os.path.join(folder, filename)
        features = extract_features(path)

        if features is None:
            print(f"  ⚠️  No pose detected: {filename}")
            skipped += 1
            continue

        rows.append(features + [label_value])

# ── Save CSV ──────────────────────────────────────────────────────
# Column names
kp_cols = []
for idx in UPPER_BODY_IDS:
    kp_cols += [f"lm{idx}_x", f"lm{idx}_y", f"lm{idx}_z"]

angle_cols = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_abduction_angle",
    "right_abduction_angle",
]

columns = kp_cols + angle_cols + ["label"]
df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

landmarker.close()

print(f"\n✅ Done!")
print(f"   Total samples  : {len(df)}")
print(f"   Correct        : {len(df[df['label']==1])}")
print(f"   Incorrect      : {len(df[df['label']==0])}")
print(f"   Skipped        : {skipped}")
print(f"   Saved to       : {OUTPUT_CSV}")
print(f"\n📐 Average angles (correct posture):")
correct = df[df['label']==1]
if len(correct) > 0:
    print(f"   Left abduction : {correct['left_abduction_angle'].mean():.1f}°")
    print(f"   Right abduction: {correct['right_abduction_angle'].mean():.1f}°")