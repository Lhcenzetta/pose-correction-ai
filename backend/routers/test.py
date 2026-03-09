# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session
# from models.user import User
# from db.database import get_db
# router = APIRouter()



# @router.get("/get_db")
# def test_home(db : Session = Depends(get_db)):
#     return db.query(User).all()

from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.get("/items/")
async def read_items(token: Annotated[str, Depends(oauth2_scheme)]):
    return {"token": token}