import pytest
from fastapi.testclient import TestClient

from cliffhanger.main import app


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def below_50k_example():
    return {
        "age": 45,
        "fnlgt": 2334,
        "education-num": 13,
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 60,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Wife",
        "race": "Black",
        "sex": "Female",
        "native-country": "Cuba",
    }


@pytest.fixture()
def above_50k_example():
    return {
        "age": 52,
        "fnlgt": 209642,
        "education-num": 9,
        "capital-gain": 123387,
        "capital-loss": 0,
        "hours-per-week": 40,
        "workclass": "Self-emp-not-inc",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
    }


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to Cliffhanger App! Let's make some predictions :)"
    }


def test_predict_below_50k(client, below_50k_example):
    response = client.post("/model/", json=below_50k_example)
    assert response.status_code == 200
    assert response.json() == [0]


def test_predict_above_50k(client, above_50k_example):
    response = client.post("/model/", json=above_50k_example)
    assert response.status_code == 200
    assert response.json() == [1]
