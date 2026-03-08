from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from models.user import User
from db.database import get_db
router = APIRouter()



@router.get("/get_db")
def test_home(db : Session = Depends(get_db)):
    return db.query(User).all()