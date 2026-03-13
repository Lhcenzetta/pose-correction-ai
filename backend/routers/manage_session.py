from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession  
from models.user import User
from models.exercices import Exercice
from models.session import Session as SessionModel  
from db.database import get_db
from schema import shcema

router = APIRouter()

@router.post("/manage_session")
def manage_session(user_session: shcema.user_session, db: DBSession = Depends(get_db)):
    find_user = db.query(User).filter(User.id == user_session.user_id).first()
    find_exercice = db.query(Exercice).filter(Exercice.id == user_session.exe_id).first()


    if not find_user or not find_exercice:
        raise HTTPException(
            status_code=404,
            detail="User not found or exercise does not exist"
        )

    start = datetime.utcnow()
    end = start + timedelta(minutes=float(user_session.duration))

    new_session = SessionModel(
        user_id=find_user.id,           
        exercice_id=find_exercice.id,  
        start_time=start,
        end_time=end,
        duration_time=user_session.duration,
        accuracy_score=0.87
    )

    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session