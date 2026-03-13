from fastapi import FastAPI
from routers import manage_session,select_exercice,authontification
from db.database import engine, Base
from models.user import User
from models.session import Session
from models.exercices import Exercice
from models.feedback import Feedback

app  = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(authontification.router)
app.include_router(manage_session.router)
app.include_router(select_exercice.router)




