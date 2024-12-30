import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        """Bewertet das Modell und gibt den MSE zurück."""
        y_pred = self.model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        return y_test, y_pred, mse

    def visualize_predictions(self, y_test, y_pred):
        """Visualisiert die tatsächlichen vs. vorhergesagten Werte."""
        plt.figure(figsize=(14, 5))
        plt.plot(y_test, label="Tatsächliche Temperatur")
        plt.plot(y_pred, label="Vorhergesagte Temperatur")
        plt.title("Tatsächliche vs. vorhergesagte Temperaturen")
        plt.xlabel("Zeit")
        plt.ylabel("Temperatur (°C)")
        plt.legend()
        plt.show()
