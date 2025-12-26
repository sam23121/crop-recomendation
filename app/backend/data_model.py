from pydantic import BaseModel
import os
# import joblib
import pickle
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
        import logging
        logger = logging.getLogger(__name__)
        
        model_file = os.path.join(os.path.dirname(__file__), "log_reg.bin")
        encoder_file = os.path.join(os.path.dirname(__file__), "label_encoder.bin")
        
        # Check if files exist
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not os.path.exists(encoder_file):
            raise FileNotFoundError(f"Encoder file not found: {encoder_file}")
        
        logger.info(f"Loading model from: {model_file}")
        logger.info(f"Loading encoder from: {encoder_file}")
        
        try:
            with open(model_file, 'rb') as f_in:
                model = pickle.load(f_in)
            
            with open(encoder_file, 'rb') as f_in:
                label_encoder = pickle.load(f_in)
            
            self.model = model
            self.label_encoder = label_encoder
            logger.info("Model and encoder loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model files: {str(e)}")
            raise

    def predict(self, input: PredictionInput):
        """Runs a prediction"""
        import logging
        logger = logging.getLogger(__name__)
        
        if not self.model or not self.label_encoder:
            raise RuntimeError("Model is not loaded. Please load the model first.")
        
        try:
            # Prepare input features
            features = [[input.nitrogen, input.potassium, input.temprature, 
                        input.humidity, input.ph, input.rainfall]]
            
            logger.debug(f"Making prediction with features: {features}")
            
            # Make prediction
            prediction = self.model.predict(features)
            
            # Transform prediction to category
            category = self.label_encoder.inverse_transform([round(prediction[0])])
            
            logger.debug(f"Prediction result: {category[0]}")
            
            return PredictionOutput(category=category[0])
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")






    