import sys
from pathlib import Path
import csv
from datetime import datetime
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Permite importar preprocess.py estando en src/
sys.path.append(str(Path(__file__).parent))
from preprocess import calcular_limites, filas_sin_atipicos

SEED = 42
TEST_SIZE = 0.2
CLEAN_PATH = "data/processed/abalone_clean.csv"
MODEL_PATH = "models/model_v1.pkl"
METRICS_PATH = "results/metrics_v1.json"
EXPERIMENTS_LOG_PATH = "results/experiments_log.csv"

NUMERIC_COLS = [
    "length", "diameter", "height", "whole_weight",
    "shucked_weight", "viscera_weight", "shell_weight",
]
FEATURES = NUMERIC_COLS + ["sex_F", "sex_I", "sex_M"]
TARGET = "age"

HIPERPARAMETROS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_leaf": 3,
    "random_state": SEED,
}

MODEL_PATH = "models/model_v2.pkl"
METRICS_PATH = "results/metrics_v2.json"

def dividir_datos(df, test_size=TEST_SIZE, seed=SEED):
    return train_test_split(df, test_size=test_size, random_state=seed)

def registrar_experimento(id_experimento, metricas):
    columnas = [
        "id_experimento", "fecha_hora", "version_modelo",
        "hiperparametros", "MAE", "RMSE", "R2",
    ]
    fila = {
        "id_experimento": id_experimento,
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "version_modelo": "v1",
        "hiperparametros": str(HIPERPARAMETROS),
        "MAE": round(metricas["MAE"], 4),
        "RMSE": round(metricas["RMSE"], 4),
        "R2": round(metricas["R2"], 4),
    }

    Path(EXPERIMENTS_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    archivo_nuevo = not Path(EXPERIMENTS_LOG_PATH).exists() or Path(EXPERIMENTS_LOG_PATH).stat().st_size == 0

    with open(EXPERIMENTS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        if archivo_nuevo:
            writer.writeheader()
        writer.writerow(fila)

def main():
    df = pd.read_csv(CLEAN_PATH)
    train, test = dividir_datos(df)

    # Límites de atípicos calculados SOLO con train
    limites = calcular_limites(train, NUMERIC_COLS)
    train = train[filas_sin_atipicos(train, limites)]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    modelo = RandomForestRegressor(**HIPERPARAMETROS)
    modelo.fit(X_train, y_train)

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH, compress=3)

    # Evaluación sobre test (sin filtrar atípicos)
    pred = modelo.predict(X_test)
    metricas = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
    }

    resultado = {
        "version": "v1",
        "modelo": "RandomForestRegressor",
        "archivo_modelo": MODEL_PATH,
        "semilla": SEED,
        "test_size": TEST_SIZE,
        "hiperparametros": HIPERPARAMETROS,
        "filas_train": len(train),
        "filas_test": len(test),
        "metricas_test": metricas,
    }

    Path(METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(json.dumps(metricas, indent=2))
    print(f"Métricas guardadas en {METRICS_PATH}")
    registrar_experimento("exp_v1_base", metricas)


if __name__ == "__main__":
    main()