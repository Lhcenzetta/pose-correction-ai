from fastapi import FastAPI
from routers import manage_session,select_exercice,authontification,feedback_session
from db.database import engine, Base
from models.user import User
from models.session import Session
from models.exercices import Exercice
from models.feedback import Feedback
from fastapi.middleware.cors import CORSMiddleware

app  = FastAPI()

Base.metadata.create_all(bind=engine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(authontification.router)
app.include_router(manage_session.router)
app.include_router(select_exercice.router)
app.include_router(feedback_session.router)




