import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)

def tune_hyperparameters(data_path, model_path):
    """
    Führt Hyperparameter-Tuning für das RandomForest-Modell durch.
    """
    try:
        data = pd.read_csv(data_path)
        X = data.drop(columns=["SalePrice"])
        y = data["SalePrice"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
        }

        logging.info("Starte Hyperparameter-Tuning...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        logging.info(f"Beste Parameter: {grid_search.best_params_}")

        predictions = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        logging.info(f"Mean Absolute Error mit besten Parametern: {mae}")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        logging.info(f"Bestes Modell erfolgreich gespeichert: {model_path}")

    except Exception as e:
        logging.error(f"Fehler beim Hyperparameter-Tuning: {e}")

if __name__ == "__main__":
    tune_hyperparameters("data/processed_house_prices.csv", "models/best_house_price_model.pkl")
