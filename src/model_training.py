import numpy as np
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

def train_and_evaluate(X, y):
    print("Iniciando Optimización de Hiperparámetros con RandomizedSearchCV...")
    start_time = time.time()
    
    # Definición del espacio de hiperparámetros
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Configuración de la validación cruzada temporal para la búsqueda
    # Usamos 3 pliegues para reducir el costo computacional local
    tscv_search = TimeSeriesSplit(n_splits=3)
    
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Inicialización del motor de búsqueda estocástica
    # n_iter=10 evaluará 10 combinaciones aleatorias del espacio de parámetros
    random_search = RandomizedSearchCV(
        estimator=rf_base, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=tscv_search, 
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Ejecución de la optimización sobre el 80% inicial para evitar mirar al futuro
    train_size = int(len(X) * 0.8)
    X_train_search, y_train_search = X.iloc[:train_size], y.iloc[:train_size]
    X_test_final, y_test_final = X.iloc[train_size:], y.iloc[train_size:]
    
    random_search.fit(X_train_search, y_train_search)
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"\nBúsqueda finalizada en {elapsed_time:.2f} minutos.")
    print("Mejores hiperparámetros encontrados:")
    print(random_search.best_params_)
    
    # Evaluación del modelo optimizado sobre el 20% no observado
    best_model = random_search.best_estimator_
    predicciones_finales = best_model.predict(X_test_final)
    mae_final = mean_absolute_error(y_test_final, predicciones_finales)
    
    print("-" * 60)
    print(f"MAE del Agente Optimizado (Test Set): {mae_final:.2f} MW")
    
    # Reentrenamiento final con el 100% de los datos usando los parámetros óptimos
    print("\nEntrenando agente final para producción con todo el dataset histórico...")
    best_model.fit(X, y)
    
    return best_model