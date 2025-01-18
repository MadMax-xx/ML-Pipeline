import pandas as pd

file_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_features.csv"


data = pd.read_csv(file_path, parse_dates=["Date Time"], index_col="Date Time")

train_size = int(len(data) * 0.8)  # 80% von 150000 für Training
train = data[:train_size]
test = data[train_size:]  # 20% für testen

output_train = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_train.csv"
output_test = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Datenanalyse Vorbereitung/jena_climate_test.csv"

train.to_csv(output_train)
test.to_csv(output_test)

print(f"Aufteilung abgeschlossen. Dateien gespeichert als:\n- {output_train}\n- {output_test}")
