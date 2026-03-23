from datetime import datetime
import os
import pickle
import threading
import numpy as np
import tensorflow as tf
from collections import deque
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
import mlflow
from models.feedback import Feedback
from routers.authentication import get_current_user
from schema.Sessionschema import SessionCreate, SessionResponse, PoseFeatures
from models.user import User
from models.exercices import Exercice
from models.session import Session as SessionModel
from db.database import get_db

router = APIRouter()


# Lazy loading of model and scaler to avoid import-time crashes and improve testability
model = None
scaler = None


def get_model_and_scaler():
    global model, scaler
    if model is None:
        model_path = os.getenv("shoulder_model")
        if model_path:
            model = tf.keras.models.load_model(model_path)
        else:
            # Fallback or error if model path is missing
            raise RuntimeError("Environment variable 'shoulder_model' is not set.")

    if scaler is None:
        scaler_path = os.getenv("scale")
        if scaler_path:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        else:
            raise RuntimeError("Environment variable 'scale' is not set.")

    return model, scaler


_pred_buffers: dict[int, deque] = {}
_session_metrics: dict[int, dict] = {}
_lock = threading.Lock()


def run_inference(session_id: int, features: list[float]):
    current_model, current_scaler = get_model_and_scaler()
    feat_scaled = current_scaler.transform(np.array(features).reshape(1, -1))
    raw_conf = float(current_model.predict(feat_scaled, verbose=0)[0][0])

    with _lock:
        if session_id not in _pred_buffers:
            _pred_buffers[session_id] = deque(maxlen=10)
        _pred_buffers[session_id].append(raw_conf)
        confidence = float(np.mean(_pred_buffers[session_id]))

    is_correct = confidence >= 0.5

    left_abd = features[26]
    right_abd = features[27]

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

    return confidence, is_correct, tip, left_abd, right_abd, raw_conf


@router.post("/session", response_model=SessionResponse)
def create_session(
    session_data: SessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not db.query(User).filter(User.id == session_data.user_id).first():
        raise HTTPException(status_code=404, detail="User not found.")
    if not db.query(Exercice).filter(Exercice.id == session_data.exercise_id).first():
        raise HTTPException(status_code=404, detail="Exercise not found.")

    new_session = SessionModel(
        user_id=session_data.user_id,
        exercise_id=session_data.exercise_id,
        start_time=datetime.utcnow(),
        duration_seconds=session_data.duration_seconds,
        status="pending",
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


@router.post("/process-pose")
def process_pose(
    body: PoseFeatures,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.query(SessionModel).filter(SessionModel.id == body.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Permission denied.")

    if session.status not in ["pending", "running", "active"]:
        return {
            "confidence": 0,
            "is_correct": False,
            "tip": f"Session is {session.status}. Processing stopped.",
            "left_abduction": 0,
            "right_abduction": 0,
            "accuracy_score": session.accuracy_score or 0,
        }

    try:
        confidence, is_correct, tip, left_abd, right_abd, raw_conf = run_inference(
            body.session_id, body.features
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    score_increment = float(confidence)

    with _lock:
        if body.session_id not in _session_metrics:
            _session_metrics[body.session_id] = {"sum": 0.0, "count": 0}

        _session_metrics[body.session_id]["sum"] += score_increment
        _session_metrics[body.session_id]["count"] += 1

        session_avg = (
            _session_metrics[body.session_id]["sum"]
            / _session_metrics[body.session_id]["count"]
        )

    session.accuracy_score = round(session_avg * 100, 2)
    session.end_time = datetime.utcnow()
    db.commit()

    return {
        "confidence": round(confidence, 4),
        "is_correct": is_correct,
        "tip": tip,
        "left_abduction": round(left_abd, 1),
        "right_abduction": round(right_abd, 1),
        "accuracy_score": round(confidence * 100, 2),
    }


def generate_feedback_comment(accuracy_score: float) -> str:
    if accuracy_score >= 90:
        return "Excellent! Your form is nearly perfect. Keep up the great work!"
    elif accuracy_score >= 75:
        return "Good job! Your form is solid. Try to maintain better alignment."
    elif accuracy_score >= 60:
        return "You're doing well! Pay attention to your posture and alignment."
    else:
        return "Keep practicing! Focus on your form. Don't hesitate to review the instructional video."


@router.post("/session/{session_id}/finalize", response_model=SessionResponse)
def finalize_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Permission denied.")

    session.status = "completed"
    session.end_time = datetime.utcnow()

    # Log to MLflow
    try:
        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment("Pose_Correction_Sessions")
        with mlflow.start_run(run_name=f"Session_{session_id}_{current_user.email}"):
            mlflow.log_param("user_id", session.user_id)
            mlflow.log_param("exercise_id", session.exercise_id)
            mlflow.log_param("duration_seconds", session.duration_seconds)
            if session.accuracy_score is not None:
                mlflow.log_metric("accuracy_score", session.accuracy_score)
    except Exception as e:
        print(f"MLflow tracking failed: {e}")

    existing = db.query(Feedback).filter(Feedback.session_id == session_id).first()
    if not existing and session.accuracy_score is not None:
        feedback = Feedback(
            session_id=session_id,
            comment=generate_feedback_comment(session.accuracy_score),
            score=int(session.accuracy_score),
            created_at=datetime.utcnow(),
        )
        db.add(feedback)

    with _lock:
        _pred_buffers.pop(session_id, None)
        _session_metrics.pop(session_id, None)

    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions/user/{user_id}")
def get_user_sessions(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied.")

    sessions = (
        db.query(SessionModel)
        .filter(SessionModel.user_id == user_id)
        .order_by(SessionModel.start_time.desc())
        .all()
    )

    result = []
    for s in sessions:
        feedback = db.query(Feedback).filter(Feedback.session_id == s.id).first()
        exercise = db.query(Exercice).filter(Exercice.id == s.exercise_id).first()
        result.append(
            {
                "id": s.id,
                "exercise_id": s.exercise_id,
                "exercise_name": (
                    exercise.name if exercise else f"Exercise #{s.exercise_id}"
                ),
                "start_time": s.start_time,
                "duration_seconds": s.duration_seconds,
                "accuracy_score": s.accuracy_score,
                "status": s.status,
                "feedback": (
                    {
                        "comment": feedback.comment,
                        "score": feedback.score,
                    }
                    if feedback
                    else None
                ),
            }
        )
    return result


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )

    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this session.",
        )

    db.delete(session)
    db.commit()

    return None
