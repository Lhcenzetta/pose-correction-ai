from sqlalchemy.orm import relationship

from db.database import Base
from sqlalchemy import Column, String, Integer, Float


class Exercice(Base):
    __tablename__ = "exercice"
    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(String)
    description = Column(String)
    duration_time = Column(Float)

    session = relationship("Session", back_populates="exercice")
