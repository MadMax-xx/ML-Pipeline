from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Modell und Feature-Namen laden
model = joblib.load("models/best_house_price_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

@app.post("/predict")
def predict(area: float, num_rooms: int, year_built: int):
    """
    API-Endpunkt zur Vorhersage von Immobilienpreisen.
    :param area: Fläche der Immobilie in Quadratmetern.
    :param num_rooms: Anzahl der Zimmer.
    :param year_built: Baujahr der Immobilie.
    :return: Vorhergesagter Preis.
    """
    # Eingabedaten mit korrekten Feature-Namen erstellen
    input_data = pd.DataFrame([[area, num_rooms, year_built]], columns=["area", "num_rooms", "year_built"])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Vorhersage
    prediction = model.predict(input_data)
    return {"predicted_price": prediction[0]}
