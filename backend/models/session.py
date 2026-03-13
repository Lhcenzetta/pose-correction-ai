from sqlalchemy import  Column, Integer ,String , Float, ForeignKey, DateTime
from db.database import Base
from sqlalchemy.orm import relationship
class Session(Base):
    __tablename__ = "session"
    id = Column(Integer, autoincrement=True , primary_key=True)
    user_id = Column(Integer , ForeignKey("users.id"))
    exercice_id = Column(Integer , ForeignKey("exercice.id"))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_time = Column(Float)
    accuracy_score = Column(Float)

    user = relationship("User" ,back_populates="session")
    exercice = relationship("Exercice", back_populates="session")
    feedbacks = relationship("Feedback", back_populates="session")