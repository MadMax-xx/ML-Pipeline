# Importieren der notwendigen Bibliotheken
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Funktionen definieren
def daten_aufteilen(X, y, test_size=0.2, random_state=42):
    """Teilt die Daten in Trainings- und Testdaten auf."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def daten_normalisieren(X_train, X_test):
    """Normalisiert die Daten (Standardisierung)."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def berechne_genauigkeit(y_test, y_pred):
    """Berechnet die Genauigkeit des Modells."""
    return accuracy_score(y_test, y_pred)

# 1. Iris-Datensatz laden
iris = load_iris()
X = iris.data  # Merkmale
y = iris.target  # Zielvariablen (Klassen)

# Datensatz beschreiben
print("Iris-Datensatz:")
print(f"Merkmale: {iris.feature_names}")
print(f"Klassen: {iris.target_names}")
print(f"Beispieldaten:\n{X[:5]}\n")

# Visualisierung der ursprünglichen Daten
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
plt.xlabel('Sepal-Länge')
plt.ylabel('Sepal-Breite')
plt.title('Iris-Datenverteilung (erste 2 Merkmale)')
plt.show()

# 2. Daten aufteilen (80% Training, 20% Test)
X_train, X_test, y_train, y_test = daten_aufteilen(X, y)

# 3. Daten normalisieren
X_train, X_test = daten_normalisieren(X_train, X_test)

# 4. Logistische Regression mit One-vs-All Strategie
log_reg = LogisticRegression(multi_class='ovr', max_iter=200)
log_reg.fit(X_train, y_train)

# 5. Vorhersagen auf den Testdaten
y_pred = log_reg.predict(X_test)

# 6. Genauigkeit berechnen
accuracy = berechne_genauigkeit(y_test, y_pred)
print(f"Genauigkeit des Modells: {accuracy:.2f}")

# 7. Detaillierter Klassifikationsbericht
print("\nKlassifikationsbericht:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 8. (Optional) Visualisierung der Entscheidungsgrenzen für 2D-Daten
# Um die Entscheidungsgrenzen zu visualisieren, reduzieren wir die Dimensionen auf 2
X_train_2D = X_train[:, :2]
X_test_2D = X_test[:, :2]

# Trainiere das Modell auf den 2D-Daten
log_reg_2D = LogisticRegression(multi_class='ovr', max_iter=200)
log_reg_2D.fit(X_train_2D, y_train)

# Meshgrid für Entscheidungsgrenzen erstellen
x_min, x_max = X_train_2D[:, 0].min() - 1, X_train_2D[:, 0].max() + 1
y_min, y_max = X_train_2D[:, 1].min() - 1, X_train_2D[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Vorhersagen für jedes Punkt im Meshgrid
Z = log_reg_2D.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot der Entscheidungsgrenzen
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=y_train, edgecolor='k', cmap=plt.cm.coolwarm)
plt.xlabel('Merkmal 1 (z.B. Sepal Länge)')
plt.ylabel('Merkmal 2 (z.B. Sepal Breite)')
plt.title('Entscheidungsgrenzen der logistischen Regression (One-vs-All)')
plt.show()