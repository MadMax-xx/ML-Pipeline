import pandas as pd

file_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_2014_2016.csv"


data = pd.read_csv(file_path, parse_dates=["Date Time"], index_col="Date Time", encoding="utf-8")


print("Datei erfolgreich geladen!") #meldung bei einer erfolreichen Hochladen der Daten
print(data.head())
print(data.info())

print("Fehlende Werte vor Bereinigung:") # sicherstellen ob alle Stellen sind gefüllt, und ob fehlden gibt
print(data.isnull().sum())

data_cleaned = data.interpolate(method='linear')#die fehlenden Wertewerden interpoliert

data_cleaned = data_cleaned[(data_cleaned["T (degC)"] > -50) & (data_cleaned["T (degC)"] < 50)]# alle physikalische unplausible Werte werden filtert (zum Beispiel Temperaturen >50°C oder < -50°C)
data_cleaned = data_cleaned[~data_cleaned.index.duplicated(keep='first')]# alle Duplikate im Index entfernen

output_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_cleaned.csv"
data_cleaned.to_csv(output_path)

print("Bereinigung abgeschlossen. Datei gespeichert als:", output_path)
