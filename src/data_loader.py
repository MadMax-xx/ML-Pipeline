import pandas as pd

class DataLoader:
    def __init__(self, url):
        self.url = url
        self.data = None

    def load_data(self):
        """Lädt die Daten von der angegebenen URL."""
        self.data = pd.read_csv(self.url, sep='\t', parse_dates=['Timestamp'])
        print("Daten erfolgreich geladen.")
        return self.data

    def reduce_frequency(self, frequency='H'):
        """Reduziert die Frequenz der Daten auf die angegebene Einheit."""
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        self.data = self.data.set_index('Timestamp').resample(frequency).mean()
        print(f"Messfrequenz reduziert auf: {frequency}")
        return self.data

    def handle_missing_values(self):
        """Behandelt fehlende Werte durch Interpolation."""
        self.data = self.data.interpolate(method='linear')
        print("Fehlende Werte behandelt.")
        return self.data
