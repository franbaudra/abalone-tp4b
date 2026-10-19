# Abalone - TP4B

Modelo simple para predecir la edad de moluscos abalone (anillos) a partir de sus mediciones físicas.

## Entorno

- **Python:** 3.11.9
- **Gestor de entornos:** venv
- **Sistema operativo de desarrollo:** Windows 11

## Instalación

```bash
git clone <url-del-repositorio>
cd abalone-tp4b
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
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