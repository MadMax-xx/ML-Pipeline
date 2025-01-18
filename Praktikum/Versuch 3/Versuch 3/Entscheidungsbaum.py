# Import der notwendigen Bibliotheken
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# 1. Daten laden und aufteilen
# Iris-Datensatz laden
X, y = load_iris(return_X_y=True)

# Aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Modell mit Standardparametern trainieren
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Testgenauigkeit berechnen
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Genauigkeit mit Standardparametern: {accuracy:.2f}")

# 3. Variation der Hyperparameter
best_accuracy = 0
best_params = None
results = []  # Liste zur Speicherung der Ergebnisse

print("\nVariation von max_depth und criterion:")
for depth in range(2, 11):
    for criterion in ['gini', 'entropy']:
        # Modell mit spezifischen Hyperparametern trainieren
        clf = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=42)
        clf.fit(X_train, y_train)

        # Testgenauigkeit berechnen
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((depth, criterion, accuracy))

        # Ausgabe der Ergebnisse
        print(f"max_depth={depth}, criterion={criterion}: Genauigkeit={accuracy:.2f}")

        # Beste Parameter speichern
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (depth, criterion)

print(f"\nBeste Konfiguration: max_depth={best_params[0]}, criterion={best_params[1]}, Genauigkeit={best_accuracy:.2f}")

# 4. Training des besten Modells
clf_best = DecisionTreeClassifier(max_depth=best_params[0], criterion=best_params[1], random_state=42)
clf_best.fit(X_train, y_train)
y_pred_best = clf_best.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_best)
print(f"Genauigkeit des besten Modells: {final_accuracy:.2f}")

# 5. Visualisierung des Entscheidungsbaums
plt.figure(figsize=(12, 8))
plot_tree(clf_best, feature_names=load_iris().feature_names, class_names=load_iris().target_names, filled=True)
plt.title("Visualisierung des besten Entscheidungsbaums")
plt.show()

# 6. Visualisierung der Genauigkeit3neu
# Ergebnisse in ein Diagramm umwandeln
depths = list(set([result[0] for result in results]))
criteria = list(set([result[1] for result in results]))

depth_accuracies = {criterion: [] for criterion in criteria}
for depth in depths:
    for criterion in criteria:
        acc = next((result[2] for result in results if result[0] == depth and result[1] == criterion), None)
        depth_accuracies[criterion].append(acc)

# Line-Plot der Genauigkeiten
plt.figure(figsize=(8, 6))
for criterion, accuracies in depth_accuracies.items():
    plt.plot(depths, accuracies, marker='o', label=f"criterion={criterion}")

plt.xlabel("max_depth")
plt.ylabel("Genauigkeit")
plt.title("Genauigkeit in Abhängigkeit von max_depth und criterion")
plt.legend(title="Kriterium")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# 7. Heatmap der Genauigkeiten
# Ergebnisse in DataFrame umwandeln
results_df = pd.DataFrame(results, columns=["max_depth", "criterion", "accuracy"])

# Pivot-Tabelle für Heatmap erstellen
heatmap_data = results_df.pivot(index="max_depth", columns="criterion", values="accuracy")

# Heatmap zeichnen
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Genauigkeit'})
plt.title("Heatmap der Genauigkeiten")
plt.xlabel("criterion")
plt.ylabel("max_depth")
plt.show()

