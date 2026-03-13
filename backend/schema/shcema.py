from pydantic import BaseModel, EmailStr
from datetime import date

class CreateUser(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str
    created_at: date

class loginUser(BaseModel):
    email :EmailStr
    password : str

class Exerciceselected(BaseModel):
    name : str
    duration_time : float
class user_session(BaseModel):
    user_id : int
    exe_id: int
    duration : float
    class config:
        orm_mode = True
