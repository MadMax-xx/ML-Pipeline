import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        """Bewertet das Modell auf dem Testdatensatz."""
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        return y_test, y_pred

    def visualize_predictions(self, y_test, y_pred):
        """Visualisiert tatsächliche vs. vorhergesagte Werte."""
        plt.plot(y_test, label='Tatsächlich')
        plt.plot(y_pred, label='Vorhersage')
        plt.legend()
        plt.title("Tatsächlich vs. Vorhersage")
        plt.show()

    def plot_loss(self, history):
        """Visualisiert Trainings- und Validierungsverluste."""
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title("Verlust über die Epochen")
        plt.show()
