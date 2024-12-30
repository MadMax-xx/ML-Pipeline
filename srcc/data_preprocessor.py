import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()

    def normalize_and_split(self, train_end="2015-12-31", val_end="2016-06-30"):
        """Normalisiert die Daten und teilt sie in Training, Validierung und Test auf."""
        # Normalisieren
        self.data = pd.DataFrame(
            self.scaler.fit_transform(self.data),
            columns=self.data.columns,
            index=self.data.index
        )
        if self.data.isnull().values.any():
            raise ValueError("Nach der Normalisierung sind noch NaN-Werte vorhanden.")
        # Aufteilen
        train_data = self.data[:train_end]
        val_data = self.data[train_end:val_end]
        test_data = self.data[val_end:]
        print(f"Daten aufgeteilt: {len(train_data)} Trainings-, {len(val_data)} Validierungs- und {len(test_data)} Testzeilen.")
        return train_data, val_data, test_data
