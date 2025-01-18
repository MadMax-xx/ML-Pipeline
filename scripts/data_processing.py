import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def preprocess_data(file_path, output_path):
    """
    Verarbeitet die Rohdaten und speichert sie als komprimierte CSV.
    :param file_path: Pfad zur Rohdaten-CSV.
    :param output_path: Pfad zur verarbeiteten CSV.
    """
    try:
        # Daten laden
        data = pd.read_csv(file_path)

        # Wichtige Spalten auswählen
        relevant_columns = [
            "LotArea", "GrLivArea", "OverallQual", "OverallCond", "YearBuilt",
            "YearRemodAdd", "TotalBsmtSF", "GarageArea", "Neighborhood",
            "HouseStyle", "SalePrice"
        ]
        data = data[relevant_columns]

        # Fehlende Werte behandeln
        data["LotArea"] = data["LotArea"].fillna(data["LotArea"].median())
        data["GrLivArea"] = data["GrLivArea"].fillna(data["GrLivArea"].median())
        data["TotalBsmtSF"] = data["TotalBsmtSF"].fillna(0)
        data["GarageArea"] = data["GarageArea"].fillna(0)

        # Numerische Normalisierung
        numeric_columns = ["LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea"]
        for col in numeric_columns:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

        # Kategorische Merkmale in numerische umwandeln
        data = pd.get_dummies(data, columns=["Neighborhood", "HouseStyle"], drop_first=True)

        # Verarbeitete Daten speichern (komprimierte CSV)
        data.to_csv(output_path, index=False, compression='zip')
        logging.info(f"Verarbeitete Daten erfolgreich gespeichert: {output_path}")
    except Exception as e:
        logging.error(f"Fehler bei der Datenverarbeitung: {e}")

if __name__ == "__main__":
    preprocess_data("data/train.csv", "data/processed_house_prices.csv.zip")
