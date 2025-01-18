import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Pfade zu Daten und Modell
file_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_features.csv"
model_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Vorhersagen/LSTM_model.keras"

# Daten laden
data = pd.read_csv(file_path, parse_dates=["Date Time"], index_col="Date Time")
print("Daten erfolgreich geladen.")

# Modell laden
try:
    model = load_model(model_path)
    print("Modell erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    exit()

# Features und Ziel definieren
features = ["T (degC)", "Temp_Lag1", "Temp_Lag2", "Temp_Rolling_Mean"]
target = "T (degC)"
data = data.dropna(subset=features + [target])

# Daten skalieren
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features + [target]])

# Sequenzen erstellen
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-1])  # Features
        y.append(data[i + sequence_length, -1])    # Ziel
    return np.array(X), np.array(y)

sequence_length = 48
X, y = create_sequences(data_scaled, sequence_length)
print(f"Sequenzen erstellt: X.shape={X.shape}, y.shape={y.shape}")

# Modell evaluieren
predictions = model.predict(X, verbose=1)

# Rücktransformation der Vorhersagen und Zielwerte
y_rescaled = scaler.inverse_transform(
    np.hstack([np.zeros((len(y), len(features))), y.reshape(-1, 1)])
)[:, -1]
predictions_rescaled = scaler.inverse_transform(
    np.hstack([np.zeros((len(predictions), len(features))), predictions])
)[:, -1]

# Metriken berechnen
mse = mean_squared_error(y_rescaled, predictions_rescaled)
mae = mean_absolute_error(y_rescaled, predictions_rescaled)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Ergebnisse speichern
results = pd.DataFrame({
    "True Values": y_rescaled,
    "Predictions": predictions_rescaled
})
results.to_csv("C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Vorhersagen/predictions.csv", index=False)
print("Vorhersagen und tatsächliche Werte gespeichert in predictions.csv.")