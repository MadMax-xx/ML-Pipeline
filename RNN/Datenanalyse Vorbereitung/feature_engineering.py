import pandas as pd

file_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_cleaned.csv"

data_cleaned = pd.read_csv(file_path, parse_dates=["Date Time"], index_col="Date Time", dayfirst=True)


data_cleaned["Month"] = data_cleaned.index.month #erstellung neue Zeitbasierte Features
data_cleaned["Day"] = data_cleaned.index.day #für alle 24std


data_cleaned["Temp_Lag1"] = data_cleaned["T (degC)"].shift(1)# neue Features Lag-Features nach dem Bericht
data_cleaned["Temp_Lag2"] = data_cleaned["T (degC)"].shift(2)


data_cleaned["Temp_Rolling_Mean"] = data_cleaned["T (degC)"].rolling(window=7).mean()# Gleitende Mittelwerte berechnen


output_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_features.csv"
data_cleaned.to_csv(output_path)     #speichern der erweiterten Daten

print(f"Feature Engineering abgeschlossen. Datei gespeichert als: {output_path}")
