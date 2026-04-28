from pydantic import BaseModel
from typing import Dict

class PredictionRequest(BaseModel):
    data: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: str