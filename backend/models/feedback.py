from sqlalchemy.orm import relationship
from db.database import Base
from sqlalchemy import Column, DateTime, String, Integer, ForeignKey

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, autoincrement=True, primary_key=True)
    session_id = Column(Integer, ForeignKey("session.id"), nullable=False)
    comment = Column(String, nullable=True)
    score = Column(Integer, nullable=True) 
    created_at = Column(DateTime, nullable=False)

    session = relationship("Session", back_populates="feedbacks")