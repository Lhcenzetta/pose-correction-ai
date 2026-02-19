import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# Setup
base_options = python.BaseOptions(model_asset_path='/Users/lait-zet/Desktop/pose-correction-ai/pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(base_options=base_options)
detector = vision.PoseLandmarker.create_from_options(options)