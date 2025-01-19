import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)

def train_model(data_path, model_path, feature_names_path):
    """
    Trainiert ein Modell zur Vorhersage von Immobilienpreisen und speichert die Feature-Namen.
    """
    try:
        logging.info("Lade verarbeitete Daten...")
        # Verarbeitete Daten laden (komprimierte CSV)
        data = pd.read_csv(data_path, compression='zip')

        # Features und Zielvariable definieren
        X = data.drop(columns=["SalePrice"])
        y = data["SalePrice"]

        logging.info("Teile die Daten in Trainings- und Testdatensätze...")
        # Datenaufteilung in Trainings- und Testdaten
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logging.info("Initialisiere und trainiere das RandomForest-Modell...")
        # RandomForest-Regressor initialisieren und trainieren
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Modell evaluieren
        logging.info("Bewerte das Modell...")
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        logging.info(f"Mean Absolute Error (MAE): {mae}")

        # Modell speichern
        logging.info("Speichere das trainierte Modell...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logging.info(f"Modell erfolgreich gespeichert: {model_path}")

        # Speichere die Feature-Namen
        joblib.dump(X.columns.tolist(), feature_names_path)
        logging.info(f"Feature-Namen erfolgreich gespeichert: {feature_names_path}")

    except Exception as e:
        logging.error(f"Fehler beim Modelltraining: {e}")

if __name__ == "__main__":
    train_model(
        "data/processed_house_prices.csv.zip",
        "models/best_house_price_model.pkl",
        "models/feature_names.pkl"
    )