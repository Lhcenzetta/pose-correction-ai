from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SessionBase(BaseModel):
    user_id: int
    exercise_id: int
    duration_seconds: Optional[float] = None

class SessionCreate(SessionBase):
    pass

class SessionUpdate(BaseModel):
    end_time: Optional[datetime] = None
    accuracy_score: Optional[float] = None
    status: Optional[str] = None

class SessionResponse(BaseModel):
    id: int
    user_id: int
    exercise_id: int
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    accuracy_score: Optional[float]
    status: str

    class Config:
        from_attributes = True

class PoseFeatures(BaseModel):
    features: list[float]        
    session_id: int 