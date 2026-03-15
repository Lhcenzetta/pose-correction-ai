from pydantic import BaseModel, EmailStr
from datetime import date, datetime

class CreateUser(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str
    created_at: date

class loginUser(BaseModel):
    email :EmailStr
    password : str

class UserResponse(CreateUser):
    id: int
    created_at: datetime

# class Exerciceselected(BaseModel):
#     name : str
#     duration_time : float
# class user_session(BaseModel):
#     user_id : int
#     exe_id: int
#     duration : float
    
# class PoseData(BaseModel):
#     features: list 

# class CreateFeedback(BaseModel):
#     session_id: int
#     comment: str
#     score: int
    
    class config:
        orm_mode = True
