from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModel:
    def __init__(self, timesteps, feature_count):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(timesteps, feature_count)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer="adam", loss="mse")
        print("LSTM-Modell erstellt.")

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Trainiert das Modell."""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        print("Modelltraining abgeschlossen.")
        return history
