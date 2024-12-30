import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_preprocess_data(self, start_date, end_date):
        """Lädt die Daten, stellt sicher, dass der Index korrekt ist, und filtert basierend auf dem Zeitbereich."""
        data = pd.read_csv(self.file_path, parse_dates=["Date Time"], index_col="Date Time")
        print(f"Zeitstempel im Datensatz: {data.index.min()} bis {data.index.max()}")

        # Sicherstellen, dass der Index ein DatetimeIndex ist
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Der Index des DataFrame ist kein DatetimeIndex.")

        # Zeitraum filtern
        if start_date < data.index.min() or end_date > data.index.max():
            raise ValueError("Der angegebene Zeitraum liegt außerhalb der verfügbaren Daten.")
        data = data[start_date:end_date]

        # Index sortieren
        data = data.sort_index()

        # Frequenz reduzieren
        data = data.resample("1H").mean()  # Reduktion auf stündliche Daten
        print("Daten erfolgreich geladen und Frequenz reduziert.")
        return data
