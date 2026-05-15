import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

def train_and_evaluate(X, y):
    print("Iniciando Validación Cruzada Temporal (Expanding Window)...")

    # Division en 5 pliegues cronologicos
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

    fold = 1
    maes = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Entrenamiento sobre la ventana expansiva
        model.fit(X_train, y_train)
        predicciones = model.predict(X_test)

        # Evaluacion
        mae = mean_absolute_error(y_test, predicciones)
        maes.append(mae)

        print(f"Fold {fold}: Train [{X_train.index.min().date()} a {X_train.index.max().date()}]"
              f"| Test [{X_test.index.min().date()} a {X_test.index.max().date()}] -> MAE: {mae:.2f} MW")
        fold += 1
        
    print("-" * 60)
    print(f"MAE Promedio de Validación Cruzada: {np.mean(maes):.2f} MW")

    # Entrenamiento del modelo final con la totalidad de los datos para produccion
    print("Entrenando agente base final con todo el dataset...")
    model.fit(X, y)

    return model