from datetime import datetime
import os
from passlib.context import CryptContext
from fastapi import APIRouter, Depends, HTTPException , status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, JWTError
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from models.user import User
from db.database import get_db
from schema import UserSchema

load_dotenv()

algorithme = "HS256"
SECRET_KEY  = os.getenv("SECRET_KEY")
barear_chema = HTTPBearer()
router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_hash_mode_pass(password):
    return pwd_context.hash(password)

def verfiy_hash_passsword(new_password , hashed_password):
    return pwd_context.verify(new_password, hashed_password)

def create_token(paylod):
    return jwt.encode(paylod, SECRET_KEY , algorithm=algorithme)

def decode_token(token):
    return jwt.decode(token , SECRET_KEY , algorithms=algorithme)

def verfiy_token(cre: HTTPAuthorizationCredentials = Depends(barear_chema)):
    token = cre.credentials
    decode = decode_token(token)
    if decode is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='this toke is invalide'
        )
    return decode

def get_current_user(token: dict = Depends(verfiy_token), db: Session = Depends(get_db)) -> User:
    email = token.get("sub")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user

@router.post("/Signup", status_code=201)
def create_account(user: UserSchema.CreateUser, db: Session = Depends(get_db)):
    exist_user = db.query(User).filter(User.email == user.email).first()
    if exist_user:
        raise HTTPException(
            status_code=400,
            detail="A user with this email already exists."
        )
    
    new_user = User(
        first_name=user.first_name,
        last_name=user.last_name,
        email=user.email,
        hashedpassword=pwd_context.hash(user.password),
        created_at=str(datetime.utcnow())
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user

@router.post("/login")
def login(user : UserSchema.loginUser ,db: Session = Depends(get_db)):
    extist_user = db.query(User).filter(User.email == user.email).first()
    if not extist_user or not pwd_context.verify(user.password, extist_user.hashedpassword):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    data={"sub": user.email}
    access_token = create_token(data)
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.get("/me", response_model=UserSchema.UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get the current user's information."""
    return current_user
