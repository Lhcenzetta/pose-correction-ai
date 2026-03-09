from fastapi import FastAPI
from routers import test
from db.database import engine, Base
from models.user import User
from models.session import Session
from models.exercices import Exercice
from models.feedback import Feedback

app  = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(test.router)




