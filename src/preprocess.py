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