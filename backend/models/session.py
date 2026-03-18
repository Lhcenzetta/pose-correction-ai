from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from db.database import Base
from sqlalchemy.orm import relationship


class Session(Base):
    __tablename__ = "session"
    id = Column(Integer, autoincrement=True, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exercise_id = Column(Integer, ForeignKey("exercice.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    accuracy_score = Column(Float, nullable=True)
    status = Column(String, default="pending", nullable=False)

    user = relationship("User", back_populates="session")
    exercice = relationship("Exercice", back_populates="session")
    feedbacks = relationship("Feedback", back_populates="session")
