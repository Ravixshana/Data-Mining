from fastapi import FastAPI, HTTPException
import numpy as np
import logging

from src.api.model_loader import load_model, load_encoder, load_features
from src.api.schema import PredictionRequest, PredictionResponse
from src.api.logger import setup_logger

# =========================
# INIT
# =========================
setup_logger()

app = FastAPI(
    title="🚬 Smoker Prediction API",
    version="1.0.0"
)

model = load_model()
encoder = load_encoder()
feature_names = load_features()


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"message": "Smoker Prediction API is running 🚀"}

# =========================
# VALIDATION FUNCTION
# =========================
def validate_input(data: dict):
    missing = [f for f in feature_names if f not in data]
    extra = [f for f in data if f not in feature_names]

    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    if extra:
        raise HTTPException(status_code=400, detail=f"Unexpected features: {extra}")


# =========================
# SINGLE PREDICTION
# =========================
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    try:
        logging.info("Received prediction request")

        input_dict = request.data

        # Validate input
        validate_input(input_dict)

        # Convert to ordered array
        input_data = np.array([input_dict[col] for col in feature_names]).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)
        label = encoder.inverse_transform(prediction)

        return PredictionResponse(prediction=str(label[0]))

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# BATCH PREDICTION (PRO LEVEL FEATURE)
# =========================
@app.post("/predict-batch")
def predict_batch(requests: list[PredictionRequest]):

    results = []

    for req in requests:
        input_data = np.array(req.features).reshape(1, -1)

        prediction = model.predict(input_data)
        label = encoder.inverse_transform(prediction)

        results.append(label[0])

    return {"predictions": results}