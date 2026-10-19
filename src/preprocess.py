import json
from datetime import date

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


def remove_invalid(df):
    return df[df["height"] > 0]


def guardar_log(filas_inicial, filas_final):
    registro = {
        "version": "v1",
        "created_at": date.today().isoformat(),
        "source": str(RAW_PATH),
        "output": str(CLEAN_PATH),
        "filas_inicial": filas_inicial,
        "filas_final": filas_final,
        "transformations": [
            "added derived feature 'age = rings + 1.5'",
            "encoded sex with one-hot encoding (sex_F, sex_I, sex_M)",
            "removed rows with height <= 0",
            "outliers: IQR x3 bounds computed on train only (applied in train_model.py, not here)",
        ],
    }
    Path("data/data_log.json").parent.mkdir(parents=True, exist_ok=True)
    with open("data/data_log.json", "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)


def main():
    df = load_raw()
    n_inicial = len(df)

    df = add_age(df)
    df = remove_invalid(df)
    df = encode_sex(df)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    print(f"Filas iniciales: {n_inicial}")
    print(f"Filas finales: {len(df)}")
    print(f"Eliminadas: {n_inicial - len(df)}")

    guardar_log(n_inicial, len(df))


if __name__ == "__main__":
    main()