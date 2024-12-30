from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModel:
    def __init__(self, input_shape, lstm_units=50, dropout_rate=0.2):
        self.model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(lstm_units),
            Dropout(dropout_rate),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        print("LSTM-Modell erstellt.")

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Trainiert das LSTM-Modell."""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        print("Training abgeschlossen.")
        return history
