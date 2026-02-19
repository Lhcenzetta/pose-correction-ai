"""
STEP 3 — Live Webcam: Shoulder Abduction Analyzer
===================================================
Shows real-time AI feedback + live arm angle measurements.

Files needed:
    shoulder_model.keras
    scaler.pkl
    pose_landmarker.task

Controls:
    Q → Quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import tensorflow as tf
from collections import deque
import threading

# ── Load model & scaler ───────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model("shoulder_model.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("✅ Model loaded!")

# ── Shared state ──────────────────────────────────────────────────
latest_landmarks = None
landmarks_lock   = threading.Lock()

# ── MediaPipe Tasks callback ──────────────────────────────────────
def on_result(result, output_image, timestamp_ms):
    global latest_landmarks
    with landmarks_lock:
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            latest_landmarks = result.pose_landmarks[0]
        else:
            latest_landmarks = None

MODEL_PATH   = "pose_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=on_result,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# ── Smoothing ─────────────────────────────────────────────────────
pred_buffer = deque(maxlen=10)

# ── Colors & constants ────────────────────────────────────────────
GREEN  = (0, 200, 0)
RED    = (0, 0, 220)
YELLOW = (0, 200, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GRAY   = (50, 50, 50)

UPPER_BODY_IDS = [11, 12, 13, 14, 15, 16, 23, 24]

# Skeleton connections to draw (upper body only)
CONNECTIONS = [
    (11, 12),  # shoulder to shoulder
    (11, 13),  # left shoulder → elbow
    (13, 15),  # left elbow → wrist
    (12, 14),  # right shoulder → elbow
    (14, 16),  # right elbow → wrist
    (11, 23),  # left shoulder → hip
    (12, 24),  # right shoulder → hip
    (23, 24),  # hip to hip
]

# ── Helper functions ──────────────────────────────────────────────
def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def get_xy(lms, idx):
    if idx < len(lms):
        return (lms[idx].x, lms[idx].y)
    return (0.0, 0.0)

def to_pixel(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

def draw_text_bg(frame, text, pos, font_scale=0.7, color=WHITE, bg=BLACK, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x-5, y-th-6), (x+tw+5, y+5), bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def draw_angle_arc(frame, center, angle, color, radius=40):
    """Draw a small arc showing the angle value near a joint."""
    cv2.ellipse(frame, center, (radius, radius), 0, 0, int(angle),
                color, 2)

def extract_features(lms):
    """Same feature extraction as Step 1 — must match exactly."""
    keypoints = []
    for idx in UPPER_BODY_IDS:
        if idx < len(lms):
            keypoints.extend([lms[idx].x, lms[idx].y, lms[idx].z])
        else:
            keypoints.extend([0.0, 0.0, 0.0])

    l_shoulder = get_xy(lms, 11)
    r_shoulder = get_xy(lms, 12)
    l_elbow    = get_xy(lms, 13)
    r_elbow    = get_xy(lms, 14)
    l_wrist    = get_xy(lms, 15)
    r_wrist    = get_xy(lms, 16)
    l_hip      = get_xy(lms, 23)
    r_hip      = get_xy(lms, 24)

    left_elbow_angle  = calc_angle(l_shoulder, l_elbow, l_wrist)
    right_elbow_angle = calc_angle(r_shoulder, r_elbow, r_wrist)
    left_abduction    = calc_angle(l_hip, l_shoulder, l_elbow)
    right_abduction   = calc_angle(r_hip, r_shoulder, r_elbow)

    angles = [left_elbow_angle, right_elbow_angle,
              left_abduction,   right_abduction]

    return keypoints + angles, left_abduction, right_abduction

# ── Main loop ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

print("🎥 Live! Press Q to quit.")
frame_ts = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)
    h, w     = frame.shape[:2]
    img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    landmarker.detect_async(mp_image, frame_ts)
    frame_ts += 1

    with landmarks_lock:
        lms = latest_landmarks

    if lms is not None:
        # ── Feature extraction ────────────────────────────────────
        features, left_abd, right_abd = extract_features(lms)
        feat_array  = np.array(features).reshape(1, -1)
        feat_scaled = scaler.transform(feat_array)

        # ── Prediction ────────────────────────────────────────────
        raw_conf = float(model.predict(feat_scaled, verbose=0)[0][0])
        pred_buffer.append(raw_conf)
        confidence = np.mean(pred_buffer)
        is_correct = confidence >= 0.5

        color = GREEN if is_correct else RED
        label = "CORRECT ✓" if is_correct else "INCORRECT ✗"

        # Specific tip based on angles
        if is_correct:
            tip = f"Good! L:{left_abd:.0f}°  R:{right_abd:.0f}°"
        else:
            if left_abd < 70 and right_abd < 70:
                tip = "Raise BOTH arms higher to 90°"
            elif left_abd < 70:
                tip = f"Raise LEFT arm more (now {left_abd:.0f}°, target 90°)"
            elif right_abd < 70:
                tip = f"Raise RIGHT arm more (now {right_abd:.0f}°, target 90°)"
            else:
                tip = "Check your arm position"

        # ── Draw skeleton (upper body only) ───────────────────────
        for start_idx, end_idx in CONNECTIONS:
            if start_idx < len(lms) and end_idx < len(lms):
                p1 = to_pixel(lms[start_idx], w, h)
                p2 = to_pixel(lms[end_idx], w, h)
                cv2.line(frame, p1, p2, color, 3)

        for idx in UPPER_BODY_IDS:
            if idx < len(lms):
                pt = to_pixel(lms[idx], w, h)
                cv2.circle(frame, pt, 6, color, -1)
                cv2.circle(frame, pt, 6, WHITE, 1)

        # ── Draw angle labels at shoulders ────────────────────────
        l_sh_pt = to_pixel(lms[11], w, h)
        r_sh_pt = to_pixel(lms[12], w, h)
        cv2.putText(frame, f"{left_abd:.0f}deg",
                    (l_sh_pt[0]-50, l_sh_pt[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        cv2.putText(frame, f"{right_abd:.0f}deg",
                    (r_sh_pt[0]+10, r_sh_pt[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

        # ── UI panel ──────────────────────────────────────────────
        # Main label
        bg = (0, 70, 0) if is_correct else (0, 0, 100)
        cv2.rectangle(frame, (10, 10), (320, 58), bg, -1)
        cv2.putText(frame, label, (18, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Confidence bar
        cv2.rectangle(frame, (10, 65), (230, 83), GRAY, -1)
        cv2.rectangle(frame, (10, 65), (10+int(confidence*220), 83), color, -1)
        draw_text_bg(frame, f"{confidence*100:.0f}% confidence",
                     (10, 105), font_scale=0.6)

        # Tip
        draw_text_bg(frame, tip, (10, h-20),
                     font_scale=0.65, color=WHITE, bg=(20, 20, 20))

    else:
        draw_text_bg(frame, "No person detected — step back",
                     (20, 50), font_scale=0.8, color=YELLOW, bg=BLACK)

    draw_text_bg(frame, "Q: Quit", (w-110, 28), font_scale=0.6)
    cv2.imshow("Shoulder Abduction Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
print("👋 Done.")