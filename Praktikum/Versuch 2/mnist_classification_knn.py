import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


mnist = datasets.fetch_openml('mnist_784')    #hier laden wir des MNIST Datensatzes
X, y = mnist.data, mnist.target.astype(np.int8)

# Aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Wahl eines Klassifikationsmodells
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluierung
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Genauigkeit auf den Testdaten: {accuracy * 100:.2f}%")

# Visualisierung von falsch klassifizierten Ziffern
incorrect = np.where(y_pred != y_test)[0]

# Umwandlung in NumPy-Array
y_test_array = y_test.to_numpy()
plt.figure(figsize=(10, 5))
for i, index in enumerate(incorrect[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test.to_numpy()[index].reshape(28, 28), cmap='gray')  # X_test in ein NumPy-Array umwandeln
    plt.title(f"Pred: {y_pred[index]}, True: {y_test_array[index]}")
    plt.axis('off')
plt.tight_layout()
plt.show()