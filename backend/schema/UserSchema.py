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