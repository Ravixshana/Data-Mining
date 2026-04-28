import joblib
import logging

def load_model():
    logging.info("Loading trained model...")
    model = joblib.load("models/best_model.pkl")
    return model

def load_encoder():
    logging.info("Loading label encoder...")
    encoder = joblib.load("models/label_encoder.pkl")
    return encoder

def load_features():
    logging.info("Loading Feature columns...")
    return joblib.load("models/features.pkl")