import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

file_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_features.csv"
data = pd.read_csv(file_path, parse_dates=["Date Time"], index_col="Date Time")

features = ["T (degC)", "Temp_Lag1", "Temp_Lag2", "Temp_Rolling_Mean"]
target = "T (degC)"

data = data.dropna(subset=features + [target]) #alle fehlenden Werte entfernen

scaler = MinMaxScaler() #MinMaxScaler für die Normalisierung
data[features + [target]] = scaler.fit_transform(data[features + [target]])

# Funktion zur Erstellung von Sequenzen (optimiert)
def create_sequences(data, feature_columns, target_column, sequence_length):
    """
    Generiert Sequenzen und Zielwerte für das LSTM-Modell.

    Parameters:
        data: DataFrame, enthält die Eingabedaten.
        feature_columns: Liste der Spaltennamen, die als Features verwendet werden.
        target_column: Zielspalte, die vorhergesagt werden soll.
        sequence_length: Länge der Sequenzen.

    Returns:
        X: NumPy-Array, enthält die Eingabesequenzen.
        y: NumPy-Array, enthält die Zielwerte.
    """
    data_array = data[feature_columns].to_numpy()
    target_array = data[target_column].to_numpy()
    X, y = [], []
    for i in range(len(data_array) - sequence_length):
        X.append(data_array[i:i + sequence_length])
        y.append(target_array[i + sequence_length])
    return np.array(X), np.array(y)

# Parameter definieren
sequence_length = 48
X, y = create_sequences(data, features, target, sequence_length)

# Daten aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modell definieren
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# Modell kompilieren
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Modell trainieren
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=32,
    verbose=1
)

# Trainingsverlauf speichern
history_df = pd.DataFrame(history.history)
history_df.to_csv("C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Vorhersagen/training_history.csv", index=False)
print("Trainingsverlauf als training_history.csv gespeichert.")

# Modell speichern
model.save("C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Vorhersagen/LSTM_model.h5")
model.save("C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Vorhersagen/LSTM_model.keras", save_format="keras")

print("Modelltraining abgeschlossen und gespeichert.")
