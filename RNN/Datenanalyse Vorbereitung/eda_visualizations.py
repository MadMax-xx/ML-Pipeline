import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_daily.csv"
data_daily = pd.read_csv(file_path, parse_dates=["Date Time"], index_col="Date Time")

# Temperaturverlauf visualisieren
plt.figure(figsize=(12, 6))
plt.plot(data_daily.index, data_daily["T (degC)"], label="Tägliche Durchschnittstemperatur", color="blue")
plt.title("Temperaturverlauf (2014-2016)")
plt.xlabel("Datum")
plt.ylabel("Temperatur (°C)")
plt.legend()
plt.show()

# Korrelationsmatrix berechnen und visualisieren
correlation_matrix = data_daily.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korrelationsmatrix der Variablen")
plt.show()

print("Korrelationsmatrix:")
print(correlation_matrix)
