# Abalone - TP4B

Modelo simple para predecir la edad de moluscos abalone (anillos) a partir de sus mediciones físicas.

## Entorno

- **Python:** 3.11.9
- **Gestor de entornos:** venv
- **Sistema operativo de desarrollo:** Windows 11

## Instalación

```bash
git clone https://github.com/franbaudra/abalone-tp4b.git
cd abalone-tp4b
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Preprocesamiento

Genera `data/processed/abalone_clean.csv` a partir de `data/raw/abalone_raw.csv` y actualiza `data/data_log.json`.

Desde la raíz del proyecto, con el entorno virtual activado:

```bash
python src/preprocess.py
```

Transformaciones aplicadas:
- Cálculo de `age = rings + 1.5`.
- Codificación one-hot de `sex`.
- Eliminación de registros con `height <= 0`.

## Entrenamiento y evaluación

Entrena un `RandomForestRegressor` para predecir `age` sobre `data/processed/abalone_clean.csv`, guarda el modelo y registra las métricas.

Desde la raíz del proyecto, con el entorno virtual activado:

```bash
python src/train_model.py
```

Qué hace el script, en orden:
1. Divide los datos en train (80%) y test (20%), con `random_state=42`.
2. Calcula los límites IQR x3 de atípicos solo sobre train y los aplica a train. El test no se filtra, para evaluar sobre la distribución real.
3. Entrena el modelo con `n_estimators=200`, `max_depth=10` y `random_state=42`.
4. Guarda el modelo en `models/model_v1.pkl` (compresión 3).
5. Calcula MAE, RMSE y R² sobre test y los guarda en `results/metrics_v1.json`.
6. Agrega una fila a `results/experiments_log.csv`.

Decisión sobre escalado: no se aplica escalado porque `RandomForestRegressor` no lo necesita. Si se usa un modelo lineal, el escalador debe ajustarse dentro de un `Pipeline` con los datos de train.

Para reproducir los resultados, ejecutar el preprocesamiento antes:

```bash
python src/preprocess.py
python src/train_model.py
```

## Resultados

**Experimento 2**

```Hiperparámetros: 

n_estimators = 300
max_depth = 15
min_samples_leaf = 3``

Resultados: 
  "MAE": 1.2345
  "RMSE": 1.6789,
  "R2": 0.5432


**Experimento 3**

Hiperparámetros: 

n_estimators = 500
max_depth = 8
min_samples_leaf = 5``

Resultados: 
  "MAE": 1.5301841058578274,
  "RMSE": 2.2110120922140513,
  "R2": 0.571577674891939
```

## Estructura del proyecto

```
abalone-tp4b/
├── data/
│   ├── raw/              # datos originales (sin modificar)
│   ├── processed/        # datos limpios o transformados
│   └── data_log.json     # registro de transformaciones (Ejercicio 2)
├── src/
│   ├── preprocess.py     # transformaciones y limpieza (Ejercicio 2)
│   └── train_model.py    # entrenamiento y evaluación (Ejercicio 3)
├── results/
│   ├── figures/          # gráficos
│   ├── metrics_v1.json   # métricas del modelo (Ejercicio 3)
│   └── experiments_log.csv  # registro de experimentos (Ejercicio 3)
├── models/               # modelos serializados (Ejercicio 3)
├── .gitignore
├── requirements.txt
└── README.md
```
