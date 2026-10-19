import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/abalone_raw.csv")
CLEAN_PATH = Path("data/processed/abalone_clean.csv")


def load_raw(path=RAW_PATH):
    return pd.read_csv(path)


def add_age(df):
    df = df.copy()
    df["age"] = df["rings"] + 1.5
    return df

def encode_sex(df):
    df = df.copy()
    dummies = pd.get_dummies(df["sex"], prefix="sex", dtype=int)
    df = pd.concat([df.drop(columns="sex"), dummies], axis=1)
    return df

def calcular_limites(df, columnas, factor=3.0):
    # Se llama SOLO con el set de entrenamiento
    limites = {}
    for col in columnas:
        q1, q3 = df[col].quantile([0.25, 0.75])
        rango = q3 - q1
        limites[col] = (q1 - factor * rango, q3 + factor * rango)
    return limites


def filas_sin_atipicos(df, limites):
    valido = pd.Series(True, index=df.index)
    for col, (minimo, maximo) in limites.items():
        valido &= df[col].between(minimo, maximo)
    return valido