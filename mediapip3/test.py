"""
VISUALIZER — See what MediaPipe detects on your images
=======================================================
Picks 3 random images from correct/ and incorrect/
and draws the skeleton + angles on them.

Run from your project folder:
    python visualize_data.py

Output: saves images to visualize_output/
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import random

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH  = "pose_landmarker.task"
DATASET_DIR = "dataset"
OUTPUT_DIR  = "visualize_output"
SAMPLES     = 3   # images to show per class

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ── Connections to draw ───────────────────────────────────────────
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]

# ── Colors ────────────────────────────────────────────────────────
COLORS = {
    "correct":   (0, 200, 0),    # Green
    "incorrect": (0, 0, 220),    # Red
}
WHITE  = (255, 255, 255)
YELLOW = (0, 220, 255)
BLACK  = (0, 0, 0)

# ── Angle helper ─────────────────────────────────────────────────
def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cosine  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def get_xy(lms, idx):
    return (lms[idx].x, lms[idx].y) if idx < len(lms) else (0.0, 0.0)

def to_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

def draw_label(img, text, pos, color=WHITE, bg=BLACK, scale=0.65, thick=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(img, (x-4, y-th-6), (x+tw+4, y+4), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thick)

# ── Process each class ────────────────────────────────────────────
for class_name, color in COLORS.items():
    folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(folder):
        print(f"⚠️  Folder not found: {folder}")
        continue

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Pick random samples
    samples = random.sample(files, min(SAMPLES, len(files)))
    print(f"\n📁 {class_name.upper()}: processing {len(samples)} samples...")

    for i, filename in enumerate(samples):
        path = os.path.join(folder, filename)
        img  = cv2.imread(path)
        if img is None:
            continue

        h, w   = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result  = landmarker.detect(mp_img)

        # ── Header bar ────────────────────────────────────────────
        header_h = 50
        header   = np.zeros((header_h, w, 3), dtype=np.uint8)
        bg_color = (0, 80, 0) if class_name == "correct" else (0, 0, 100)
        header[:] = bg_color
        label_text = f"{'✓ CORRECT' if class_name == 'correct' else '✗ INCORRECT'}  —  {filename}"
        cv2.putText(header, label_text, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        img = np.vstack([header, img])
        h   = img.shape[0]  # update height after adding header

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            draw_label(img, "⚠ No pose detected", (10, h-20),
                       color=(0,165,255), bg=BLACK)
            out_path = os.path.join(OUTPUT_DIR, f"{class_name}_{i+1}_NO_POSE.jpg")
            cv2.imwrite(out_path, img)
            print(f"  ⚠ No pose: {filename}")
            continue

        lms = result.pose_landmarks[0]

        # ── Draw skeleton ─────────────────────────────────────────
        for s, e in CONNECTIONS:
            if s < len(lms) and e < len(lms):
                p1 = (int(lms[s].x * w), int(lms[s].y * (h - header_h)) + header_h)
                p2 = (int(lms[e].x * w), int(lms[e].y * (h - header_h)) + header_h)
                cv2.line(img, p1, p2, color, 3)

        for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
            if idx < len(lms):
                px = int(lms[idx].x * w)
                py = int(lms[idx].y * (h - header_h)) + header_h
                cv2.circle(img, (px, py), 7, color, -1)
                cv2.circle(img, (px, py), 7, WHITE, 1)

        # ── Compute & display angles ───────────────────────────────
        l_sh  = get_xy(lms, 11)
        r_sh  = get_xy(lms, 12)
        l_el  = get_xy(lms, 13)
        r_el  = get_xy(lms, 14)
        l_wr  = get_xy(lms, 15)
        r_wr  = get_xy(lms, 16)
        l_hip = get_xy(lms, 23)
        r_hip = get_xy(lms, 24)

        left_abd  = calc_angle(l_hip,  l_sh, l_el)
        right_abd = calc_angle(r_hip,  r_sh, r_el)
        left_elb  = calc_angle(l_sh,   l_el, l_wr)
        right_elb = calc_angle(r_sh,   r_el, r_wr)

        # Draw angle text near each shoulder
        def lm_px(idx):
            return (int(lms[idx].x * w),
                    int(lms[idx].y * (h - header_h)) + header_h)

        if 11 < len(lms):
            draw_label(img, f"Abd: {left_abd:.0f}deg",
                       (lm_px(11)[0]-80, lm_px(11)[1]-15), color=YELLOW)
        if 12 < len(lms):
            draw_label(img, f"Abd: {right_abd:.0f}deg",
                       (lm_px(12)[0]+10, lm_px(12)[1]-15), color=YELLOW)
        if 13 < len(lms):
            draw_label(img, f"{left_elb:.0f}deg",
                       (lm_px(13)[0]-20, lm_px(13)[1]+25), color=WHITE, scale=0.5)
        if 14 < len(lms):
            draw_label(img, f"{right_elb:.0f}deg",
                       (lm_px(14)[0]+10, lm_px(14)[1]+25), color=WHITE, scale=0.5)

        # ── Info panel at bottom ───────────────────────────────────
        info = f"L abduction:{left_abd:.1f}deg  R abduction:{right_abd:.1f}deg  L elbow:{left_elb:.1f}deg  R elbow:{right_elb:.1f}deg"
        draw_label(img, info, (10, h - 12), color=WHITE, bg=(30,30,30), scale=0.55)

        # ── Save ──────────────────────────────────────────────────
        out_path = os.path.join(OUTPUT_DIR, f"{class_name}_{i+1}.jpg")
        cv2.imwrite(out_path, img)
        print(f"  ✅ Saved: {out_path}")
        print(f"     L_abduction={left_abd:.1f}°  R_abduction={right_abd:.1f}°")

landmarker.close()

print(f"\n🎉 Done! Check the '{OUTPUT_DIR}/' folder for your visualizations.")
print("Share what you see and we'll fix the angles if needed!")