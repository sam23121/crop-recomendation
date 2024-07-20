from pydantic import BaseModel
import os
import joblib
import pickle
from fastapi import Depends, FastAPI
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder



class PredictionInput(BaseModel):
    nitrogen: float 
    potassium: float 
    temprature: float 
    humidity: float 
    ph: float
    rainfall: float

class PredictionOutput(BaseModel):
    category: str



class CropGroupsModel:
    model: LogisticRegression | None = None
    label_encoder: LabelEncoder | None = None
    # targets: list[str] | None = None

    def load_model(self) -> None:
        """Loads the model"""
        model_file = os.path.join(os.path.dirname(__file__), "log_reg.bin")
        encoder_file = os.path.join(os.path.dirname(__file__), "label_encoder.bin")
        # loaded_model: tuple[Pipeline, list[str]] = joblib.load(model_file)
        with open(model_file, 'rb') as f_in:
            model = pickle.load(f_in)
        
        with open(encoder_file, 'rb') as f_in:
            label_encoder = pickle.load(f_in)
        
        self.model = model
        self.label_encoder = label_encoder

    def predict(self, input: PredictionInput):
        """Runs a prediction"""
        if not self.model or not self.label_encoder:
            raise RuntimeError("Model is not loaded")
        
        prediction = self.model.predict([[input.nitrogen, input.potassium, input.temprature, 
                                          input.humidity, input.ph, input.rainfall]])
        
        category = self.label_encoder.inverse_transform([round(prediction[0])])
        
        return PredictionOutput(category=category[0])






    