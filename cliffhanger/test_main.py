import pytest
from fastapi.testclient import TestClient

from cliffhanger.main import app


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def below_50k_example():
    return {
      "age": 39,
      "fnlgt": 77516,
      "education_num": 13,
      "capital_gain": 2174,
      "capital_loss": 0,
      "hours_per_week": 40,
      "workclass": "State-gov",
      "education": "Bachelors",
      "marital_status": "Never-married",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "native_country": "United-States"
    }


@pytest.fixture()
def above_50k_example():
    return {
        "age": 52,
        "fnlgt": 209642,
        "education_num": 9,
        "capital_gain": 123387,
        "capital_loss": 0,
        "hours_per_week": 40,
        "workclass": "Self-emp-not-inc",
        "education": "Bachelors",
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States"
    }


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == \
           {"message": "Welcome to Cliffhanger! Let's make some predictions :)"}


def test_predict_below_50k(client, below_50k_example):
    response = client.post("/model/", json=below_50k_example)
    assert response.status_code == 200
    assert response.json() == 0


def test_predict_above_50k(client, above_50k_example):
    response = client.post("/model/", json=above_50k_example)
    assert response.status_code == 200
    assert response.json() == 1
