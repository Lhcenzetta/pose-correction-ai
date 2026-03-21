from pydantic import BaseModel
from typing import Optional


class ExerciseBase(BaseModel):
    name: str
    description: Optional[str] = None
    duration_time: Optional[float] = None


class ExerciseCreate(ExerciseBase):
    pass


class ExerciseResponse(ExerciseBase):
    id: int
