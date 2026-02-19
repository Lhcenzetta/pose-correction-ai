import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque




# Load trained model
model = load_model("/Users/lait-zet/Desktop/pose-correction-ai/model/pose_correction_model.h5")

IMG_SIZE = 128

# Open webcam
cap = cv2.VideoCapture(0)

correct_frames = 0
total_frames = 0

prediction_buffer = deque(maxlen=10)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for model
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)
    probability = prediction[0][0]

    prediction_buffer.append(probability)

    # Average last predictions
    smoothed_prob = np.mean(prediction_buffer)

    total_frames += 1

    if smoothed_prob > 0.5:
        label = "CORRECT"
        color = (0, 255, 0)
        correct_frames += 1
    else:
        label = "INCORRECT"
        color = (0, 0, 255) 

    # Display text
    cv2.putText(frame, f"{label} ({probability:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2)

    cv2.imshow("Pose Correction AI", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final Score
if total_frames > 0:
    score = (correct_frames / total_frames) * 100
    if score >= 80:
        print(f"Great job! Your score: {score:.2f}%")
    elif score >= 50:
        print(f"Good effort! Your score: {score:.2f}%")
    else:
        print(f"Needs improvement. Your score: {score:.2f}%")
