import pandas as pd


file_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_2014_2016.csv"
data = pd.read_csv(file_path, parse_dates=["Date Time"], index_col="Date Time", encoding="utf-8")   #ich hatte Probleme bei Laden der Datei, hab aber eine Lösung mit Notepad++ Encding gefunden durch UTF-()

print("Datei erfolgreich geladen!")
print(data.head())
print(data.info())
