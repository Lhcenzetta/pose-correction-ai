from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from db.database import get_db
from models.feedback import Feedback
from models.session import Session as SessionModel
from schema.Feedbackschema import FeedbackCreate, FeedbackUpdate, FeedbackResponse
from routers.authontification import get_current_user

router = APIRouter()


def generate_feedback_comment(accuracy_score: float) -> str:
    if accuracy_score >= 90:
        return "Excellent! Your form is nearly perfect. Keep up the great work!"
    elif accuracy_score >= 75:
        return "Good job! Your form is solid. Try to maintain better alignment for even better results."
    elif accuracy_score >= 60:
        return "You're doing well! Pay attention to your posture and try to maintain better alignment."
    else:
        return "Keep practicing! Focus on your form and alignment. Don't hesitate to review the instructional video."


@router.post("/Feedback", response_model=FeedbackResponse)
def create_feedback(
    feedback_data: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    session = (
        db.query(SessionModel)
        .filter(SessionModel.id == feedback_data.session_id)
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )

    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to create feedback for this session.",
        )

    comment = feedback_data.comment
    if not comment and session.accuracy_score is not None:
        comment = generate_feedback_comment(session.accuracy_score)

    score = feedback_data.score
    if score is None and session.accuracy_score is not None:
        score = int(session.accuracy_score)

    new_feedback = Feedback(
        session_id=feedback_data.session_id,
        comment=comment,
        score=score,
        created_at=datetime.utcnow(),
    )

    db.add(new_feedback)
    db.commit()
    db.refresh(new_feedback)

    return new_feedback


@router.get("/session/{session_id}", response_model=List[FeedbackResponse])
def get_session_feedback(
    session_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
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
            detail="You do not have permission to view feedback for this session.",
        )

    feedbacks = db.query(Feedback).filter(Feedback.session_id == session_id).all()
    return feedbacks


@router.get("/{feedback_id} get_feedback", response_model=FeedbackResponse)
def get_feedback(
    feedback_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found.",
        )

    session = (
        db.query(SessionModel).filter(SessionModel.id == feedback.session_id).first()
    )
    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this feedback.",
        )

    return feedback


@router.delete("/{feedback_id}delete_feedback", status_code=status.HTTP_204_NO_CONTENT)
def delete_feedback(
    feedback_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found.",
        )

    session = (
        db.query(SessionModel).filter(SessionModel.id == feedback.session_id).first()
    )
    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this feedback.",
        )

    db.delete(feedback)
    db.commit()

    return None
