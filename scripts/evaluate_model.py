import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)

def evaluate_model(data_path, model_path, feature_path, results_dir="results"):
    """
    Bewertet das trainierte Modell und visualisiert die Ergebnisse.
    """
    try:
        # Erstelle das Verzeichnis für Ergebnisse, falls nicht vorhanden
        os.makedirs(results_dir, exist_ok=True)

        # Lade die CSV-Datei
        data = pd.read_csv(data_path)

        # Sicherstellen, dass die Features mit dem Training übereinstimmen
        feature_names = joblib.load(feature_path)
        if not set(feature_names).issubset(data.columns):
            raise ValueError("Feature-Namen stimmen nicht mit den Daten überein")

        X = data[feature_names]
        y_true = data["SalePrice"]

        # Lade das Modell und mache Vorhersagen
        model = joblib.load(model_path)
        y_pred = model.predict(X)

        # Berechne die Evaluationsmetriken
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        logging.info(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

        # Echte vs. Vorhergesagte Werte visualisieren
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", label="Ideal")
        plt.xlabel("Echte Preise")
        plt.ylabel("Vorhergesagte Preise")
        plt.title("Echte vs. Vorhergesagte Preise")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(results_dir, "true_vs_predicted.png"))
        plt.show()

        # Residuenplot erstellen
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Vorhergesagte Preise")
        plt.ylabel("Residuen")
        plt.title("Residuenplot")
        plt.grid()
        plt.savefig(os.path.join(results_dir, "residuals.png"))
        plt.show()

    except Exception as e:
        logging.error(f"Fehler bei der Evaluierung: {e}")

if __name__ == "__main__":
    evaluate_model(
        "data/processed_house_prices.csv",
        "models/best_house_price_model.pkl",
        "models/feature_names.pkl"
    )
