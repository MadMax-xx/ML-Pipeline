import pytest
import pandas as pd
from src.data_loader import DataLoader

def test_load_data():
    loader = DataLoader("test_url")
    assert loader is not None

def test_reduce_frequency():
    loader = DataLoader("test_url")
    loader.data = pd.DataFrame({
        "Timestamp": pd.date_range("2023-01-01", periods=10, freq="T"),
        "Value": range(10)
    })
    reduced_data = loader.reduce_frequency("H")
    assert len(reduced_data) == 1
