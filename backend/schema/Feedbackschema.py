from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class FeedbackBase(BaseModel):
    comment: Optional[str] = None
    score: Optional[int] = None

class FeedbackCreate(FeedbackBase):
    session_id: int

class FeedbackUpdate(FeedbackBase):
    pass

class FeedbackResponse(BaseModel):
    id: int
    session_id: int
    comment: Optional[str]
    score: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True
