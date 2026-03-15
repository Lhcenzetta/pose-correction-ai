from datetime import datetime, timedelta
import pickle
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException,status
import numpy as np
from sqlalchemy.orm import Session as DBSession
import tensorflow as tf  
from routers.authontification import get_current_user
from schema.Sessionschema import PoseData, SessionCreate, SessionResponse
from models.user import User
from models.exercices import Exercice
from models.session import Session as SessionModel  
from db.database import get_db

import os
router = APIRouter()

shoulder = os.getenv("shoulder_model")
scal = os.getenv("scale")
model = tf.keras.models.load_model(shoulder)
with open(scal , "rb") as f:
    scaler = pickle.load(f)



def calculate_accuracy_score(pose_data: PoseData) -> float:
    """
    Calculate accuracy score from pose data using AI model.
    This is a placeholder function that should be replaced with actual AI inference.
    """
    # TODO: Implement actual AI model inference
    # For now, return a random score between 0 and 100
    return float(np.random.uniform(60, 100))

@router.post("/session", response_model=SessionResponse)
def create_session(
    session_data: SessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = db.query(User).filter(User.id == session_data.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    
    
    exercise = db.query(Exercice).filter(Exercice.id == session_data.exercise_id).first()
    if not exercise:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Exercise not found.",
        )
    
    start = datetime.utcnow()

    new_session = SessionModel(
        user_id=session_data.user_id,
        exercise_id=session_data.exercise_id,
        start_time=start,
        duration_seconds=session_data.duration_seconds,
        status="pending"
    )
    
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    return new_session



@router.post("/{session_id}/process-pose", response_model=SessionResponse)
def process_pose_data(
    session_id: int,
    pose_data: PoseData,
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
            detail="You do not have permission to update this session.",
        )
    
   
    accuracy_score = calculate_accuracy_score(pose_data)
    
    session.accuracy_score = accuracy_score
    session.status = "completed"
    session.end_time = datetime.utcnow()
    
    db.commit()
    db.refresh(session)
    
    return session



@router.get("/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific session by ID."""
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )
    
    # Verify ownership
    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this session.",
        )
    
    return session




@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a session."""
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )
    
    # Verify ownership
    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this session.",
        )
    
    db.delete(session)
    db.commit()
    
    return None
