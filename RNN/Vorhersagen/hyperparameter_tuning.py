import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Daten vorbereiten
file_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_features.csv"
data = pd.read_csv(file_path, parse_dates=["Date Time"], index_col="Date Time")

features = ["T (degC)", "Temp_Lag1", "Temp_Lag2", "Temp_Rolling_Mean"]
target = "T (degC)"
data = data.dropna(subset=features + [target])

# Normalisierung
scaler = MinMaxScaler()
data[features + [target]] = scaler.fit_transform(data[features + [target]])

# Sequenzen erstellen
def create_sequences(data, feature_columns, target_column, sequence_length):
    data_array = data[feature_columns].to_numpy()
    target_array = data[target_column].to_numpy()
    X, y = [], []
    for i in range(len(data_array) - sequence_length):
        X.append(data_array[i:i + sequence_length])
        y.append(target_array[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 48
X, y = create_sequences(data, features, target, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Hyperparameter-Tuning
best_model = None
best_mae = float("inf")
results = []

for lstm_units in [32, 64, 128]:
    for lr in [0.001, 0.01, 0.1]:
        for batch_size in [16, 32, 64]:
            print(f"Training mit LSTM-Einheiten: {lstm_units}, Lernrate: {lr}, Batch-Größe: {batch_size}")
            model = Sequential([
                LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                epochs=20, batch_size=batch_size, verbose=0)
            mae = min(history.history['val_mae'])
            results.append((lstm_units, lr, batch_size, mae))
            print(f"MAE: {mae}")
            if mae < best_mae:
                best_mae = mae
                best_model = model

# Ergebnisse dokumentieren
results_df = pd.DataFrame(results, columns=["LSTM-Einheiten", "Lernrate", "Batch-Größe", "MAE"])
results_df.to_csv("C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Vorhersagen/hyperparameter_results.csv", index=False)
print("Ergebnisse gespeichert in hyperparameter_results.csv.")

# Bestes Modell speichern
print("Bestes Modell erreicht MAE:", best_mae)
best_model.save("C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Vorhersagen/Best_LSTM_Model.keras")
