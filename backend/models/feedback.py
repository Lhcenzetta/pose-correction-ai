from sqlalchemy.orm import relationship
from db.database import Base
from sqlalchemy import Column, String, Integer, ForeignKey

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("session.id"))
    comment = Column(String)
    score = Column(Integer)

    session = relationship("Session", back_populates="feedbacks")