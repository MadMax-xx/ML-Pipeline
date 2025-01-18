import os
import pandas as pd

# Absoluter Pfad zur bereinigten Datei
file_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_cleaned.csv"

# Datei laden
data_cleaned = pd.read_csv(file_path, parse_dates=["Date Time"], index_col="Date Time", encoding="utf-8")

# Sicherstellen, dass der Index ein DatetimeIndex ist
data_cleaned.index = pd.to_datetime(data_cleaned.index, errors="coerce")

# Überprüfen auf doppelte Zeitstempel und entfernen
data_cleaned = data_cleaned[~data_cleaned.index.duplicated()]

# Lücken im Zeitindex auffüllen, basierend auf 10-Minuten-Intervallen
data_cleaned = data_cleaned.asfreq('10min')

# Fehlende Werte nach dem Auffüllen interpolieren
data_cleaned = data_cleaned.interpolate(method='linear')

# Auswahl nur numerischer Spalten für die Aggregation
numeric_columns = data_cleaned.select_dtypes(include="number").columns

# Reduktion auf stündliche Mittelwerte
data_hourly = data_cleaned[numeric_columns].resample('h').mean()

# Reduktion auf tägliche Mittelwerte
data_daily = data_cleaned[numeric_columns].resample('d').mean()

# Sicherstellen, dass alle Tage im täglichen Datensatz vorhanden sind
data_daily = data_daily.asfreq('d')

# Verzeichnis sicherstellen
output_dir = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung"
os.makedirs(output_dir, exist_ok=True)

# Ergebnisse speichern
output_hourly = os.path.join(output_dir, "jena_climate_hourly.csv")
output_daily = os.path.join(output_dir, "jena_climate_daily.csv")

data_hourly.to_csv(output_hourly)
data_daily.to_csv(output_daily)

print(f"Reduktion abgeschlossen. Dateien gespeichert als:\n- {output_hourly}\n- {output_daily}")
