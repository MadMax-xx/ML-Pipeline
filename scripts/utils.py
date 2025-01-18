import pandas as pd
import logging
import joblib
import os

def configure_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def load_and_prepare_data(data_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]
    return X, y

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

def load_model(model_path):
    return joblib.load(model_path)
