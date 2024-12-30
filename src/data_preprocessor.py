import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()
        self.features = None
        self.labels = None

    def select_features_and_labels(self, feature_columns, target_column):
        """Wählt relevante Features und das Ziel aus."""
        self.features = self.data[feature_columns]
        self.labels = self.data[target_column].shift(-1)
        self.data = self.data.dropna()
        print("Features und Labels ausgewählt.")
        return self.features, self.labels

    def normalize_data(self):
        """Normalisiert die Daten mit MinMaxScaler."""
        self.data = self.scaler.fit_transform(self.data)
        print("Daten normalisiert.")
        return self.data

    def create_sequences(self, timesteps):
        """Erstellt Sequenzen für das Modell."""
        X, y = [], []
        for i in range(len(self.data) - timesteps):
            X.append(self.data[i:i+timesteps, :-1])
            y.append(self.data[i+timesteps, -1])
        print("Sequenzen erstellt.")
        return np.array(X), np.array(y)
