# test_main.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.frontend.app import app, CropGroupsModel, PredictionInput, PredictionOutput

# Create a test client
client = TestClient(app)

# Mock data
test_input = PredictionInput(
    nitrogen=10.0,
    potassium=10.0,
    temprature=10.0,
    humidity=10.0,
    ph=7.0,
    rainfall=10.0

)


expected_output = PredictionOutput(
    category="papaya",
)

@pytest.fixture(autouse=True)
def mock_crop_groups_model():
    with patch.object(CropGroupsModel, 'load_model') as mock_load_model:
        with patch.object(CropGroupsModel, 'predict', return_value=expected_output) as mock_predict:
            yield mock_predict

def test_prediction_endpoint(mock_crop_groups_model):
    response = client.post("/prediction", json=test_input.model_dump())
    assert response.status_code == 200
    assert response.json() == expected_output.model_dump()
