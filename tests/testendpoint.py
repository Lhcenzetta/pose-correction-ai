# tests/test_endpoints.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock

from main import app
from db.database import get_db, Base
from models.user import User


SQLALCHEMY_TEST_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_TEST_URL,
    connect_args={"check_same_thread": False}  # needed for SQLite only
)
TestingSession = sessionmaker(bind=engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db


client = TestClient(app)



@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.create_all(bind=engine)    # create tables
    yield                                     # run the test
    Base.metadata.drop_all(bind=engine)      # clean up after



def test_signup_success():
    response = client.post("/Signup", json={
        "first_name": "Lahcen",
        "last_name": "Test",
        "email": "lahcen@test.com",
        "password": "secret123"
    })
    # Assert means "check that this is true"
    assert response.status_code == 201
    assert response.json()["email"] == "lahcen@test.com"



def test_signup_duplicate_email():
    # First signup
    client.post("/Signup", json={
        "first_name": "Lahcen",
        "last_name": "Test",
        "email": "lahcen@test.com",
        "password": "secret123"
    })
    # Second signup with same email — should fail
    response = client.post("/Signup", json={
        "first_name": "Other",
        "last_name": "User",
        "email": "lahcen@test.com",
        "password": "other123"
    })
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]



def test_login_success():
    # First create the user
    client.post("/Signup", json={
        "first_name": "Lahcen",
        "last_name": "Test",
        "email": "lahcen@test.com",
        "password": "secret123"
    })
    # Then login
    response = client.post("/login", json={
        "email": "lahcen@test.com",
        "password": "secret123"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()     # token exists
    assert response.json()["token_type"] == "bearer"



def test_login_wrong_password():
    client.post("/Signup", json={
        "first_name": "Lahcen",
        "last_name": "Test",
        "email": "lahcen@test.com",
        "password": "secret123"
    })
    response = client.post("/login", json={
        "email": "lahcen@test.com",
        "password": "WRONGPASSWORD"
    })
    assert response.status_code == 401



def test_get_me():
    # Create + login to get a real token
    client.post("/Signup", json={
        "first_name": "Lahcen",
        "last_name": "Test",
        "email": "lahcen@test.com",
        "password": "secret123"
    })
    login = client.post("/login", json={
        "email": "lahcen@test.com",
        "password": "secret123"
    })
    token = login.json()["access_token"]

    # Use that token in the Authorization header
    response = client.get("/me", headers={
        "Authorization": f"Bearer {token}"
    })
    assert response.status_code == 200
    assert response.json()["email"] == "lahcen@test.com"



def test_get_me_no_token():
    response = client.get("/me")
    assert response.status_code in [401, 403]