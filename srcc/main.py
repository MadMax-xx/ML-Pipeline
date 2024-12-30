import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from lstm_model import LSTMModel
from model_evaluator import ModelEvaluator
import numpy as np

# Sequenzen erstellen
def create_sequences(data, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps, 0])  # Ziel: Temperatur
    return np.array(X), np.array(y)

def main():
    file_path = "data/jena_climate_2014_to_2016.csv"
    try:
        loader = DataLoader(file_path)
        data = loader.load_and_preprocess_data(start_date="2014-01-01 00:10:00", end_date="2016-12-31 23:50:00")
    except ValueError as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return

    # Daten vorbereiten
    preprocessor = DataPreprocessor(data)
    try:
        train_data, val_data, test_data = preprocessor.normalize_and_split()
    except ValueError as e:
        print(f"Fehler bei der Datenvorverarbeitung: {e}")
        return

    # Sequenzen erstellen
    timesteps = 24
    X_train, y_train = create_sequences(train_data.values, timesteps)
    X_val, y_val = create_sequences(val_data.values, timesteps)
    X_test, y_test = create_sequences(test_data.values, timesteps)

    # LSTM-Modell erstellen und trainieren
    model = LSTMModel(timesteps, train_data.shape[1])
    try:
        history = model.train(X_train, y_train, X_val, y_val)
    except Exception as e:
        print(f"Fehler beim Modelltraining: {e}")
        return

    # Modell evaluieren
    evaluator = ModelEvaluator(model.model)
    try:
        y_test, y_pred, mse = evaluator.evaluate(X_test, y_test)
        evaluator.visualize_predictions(y_test, y_pred)
    except Exception as e:
        print(f"Fehler bei der Modellbewertung: {e}")
        return

if __name__ == "__main__":
    main()
