import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)

def evaluate_model(data_path, model_path):
    """
    Bewertet das trainierte Modell und visualisiert die Ergebnisse.
    """
    try:
        data = pd.read_csv(data_path, compression='zip')
        X = data.drop(columns=["SalePrice"])
        y_true = data["SalePrice"]

        model = joblib.load(model_path)
        y_pred = model.predict(X)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        logging.info(f"MAE: {mae}, MSE: {mse}, R²: {r2}")

        results_dir = "results/"
        os.makedirs(results_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
        plt.xlabel("Echte Preise")
        plt.ylabel("Vorhergesagte Preise")
        plt.title("Echte vs. Vorhergesagte Preise")
        plt.savefig(f"{results_dir}/true_vs_predicted.png")
        plt.close()

        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Vorhergesagte Preise")
        plt.ylabel("Residuen")
        plt.title("Residuenplot")
        plt.savefig(f"{results_dir}/residuals.png")
        plt.close()

    except Exception as e:
        logging.error(f"Fehler bei der Evaluierung: {e}")

if __name__ == "__main__":
    evaluate_model("data/processed_house_prices.csv.zip", "models/best_house_price_model.pkl")
