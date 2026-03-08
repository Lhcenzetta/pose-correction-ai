from fastapi import FastAPI
from routers import test
from db.database import engine, Base

app  = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(test.router)




