from sqlalchemy.orm import relationship

from db.database import Base
from sqlalchemy import Column , String , Integer, Float

class User(Base):
     __tablename__ = "users"
     id = Column(Integer, autoincrement=True , primary_key=True)
     first_name = Column(String)
     last_name = Column(String)
     email = Column(String, unique=True)
     hashedpassword = Column(String)
     created_at = Column(String)

     session = relationship("Session", back_populates="user")