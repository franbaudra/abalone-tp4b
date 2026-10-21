import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
CLEAN_PATH = "data/processed/abalone_clean.csv"


def dividir_datos(df, test_size=TEST_SIZE, seed=SEED):
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train, test


def main():
    df = pd.read_csv(CLEAN_PATH)
    train, test = dividir_datos(df)

    print(f"Filas totales: {len(df)}")
    print(f"Train: {len(train)}")
    print(f"Test: {len(test)}")


if __name__ == "__main__":
    main()