import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# **Pfade zu den Dateien**
training_history_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Vorhersagen/training_history.csv"
hyperparameter_results_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Vorhersagen/hyperparameter_results.csv"
predictions_path = "C:/Users/bilal/OneDrive - Hochschule Düsseldorf/Desktop/HSD/ML/Projektsarbeit/ml-ws2425-team6-2/RNN/Vorhersagen/predictions.csv"

# **1. Verlustkurven visualisieren**
training_history = pd.read_csv(training_history_path)

plt.figure(figsize=(10, 6))
plt.plot(training_history['loss'], label="Trainingsverlust", color='blue')
plt.plot(training_history['val_loss'], label="Validierungsverlust", color='orange')
plt.title("Verlustkurve während des Trainings")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# **2. Tatsächliche vs. vorhergesagte Werte (Linienplot)**
predictions = pd.read_csv(predictions_path)
y_true = predictions['True Values']
y_pred = predictions['Predictions']

plt.figure(figsize=(14, 7))
plt.plot(y_true, label="Tatsächliche Werte", color='blue')
plt.plot(y_pred, label="Vorhergesagte Werte", color='orange')
plt.title("Tatsächliche vs. vorhergesagte Temperaturen")
plt.xlabel("Zeitpunkte")
plt.ylabel("Temperatur (°C)")
plt.legend()
plt.grid()
plt.show()

# **3. Scatterplot der tatsächlichen vs. vorhergesagten Werte**
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label="Perfekte Vorhersage")
plt.title("Scatterplot: Tatsächliche vs. vorhergesagte Werte")
plt.xlabel("Tatsächliche Werte")
plt.ylabel("Vorhergesagte Werte")
plt.legend()
plt.grid()
plt.show()

# **4. Ergebnisse des Hyperparameter-Tunings**
hyperparameter_results = pd.read_csv(hyperparameter_results_path)
print("Spaltennamen im Hyperparameter-Ergebnis-Datensatz:", hyperparameter_results.columns)

plt.figure(figsize=(12, 8))
sns.barplot(
    x="LSTM Units",
    y="Validation MAE",
    hue="Learning Rate",
    data=hyperparameter_results,
    palette="viridis"
)
plt.title("Einfluss von Hyperparametern auf die Validation MAE")
plt.xlabel("LSTM Units")
plt.ylabel("Validation MAE")
plt.legend(title="Learning Rate")
plt.grid()
plt.show()

# **5. Paarweise Beziehungen zwischen Hyperparametern**
sns.pairplot(hyperparameter_results, diag_kind='kde')
plt.suptitle("Paarweise Beziehungen zwischen Hyperparametern und Validation MAE", y=1.02)
plt.show()

# **6. Berechnung von MSE und MAE**
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
