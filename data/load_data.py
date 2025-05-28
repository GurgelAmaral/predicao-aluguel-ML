import pandas as pd

def load_data(path="data\sao-paulo-properties-april-2019.csv"):
    return pd.read_csv(path)

