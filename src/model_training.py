import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import time
import warnings

# Ocultar advertencias para mantener la consola limpia
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def train_and_evaluate(X, y):
    print("Iniciando Optimización Bayesiana de Hiperparámetros con Optuna y XGBoost...")
    start_time = time.time()
    
    # Separación cronológica: 80% para Optuna, 20% para el Test Final
    train_size = int(len(X) * 0.8)
    X_train_search, y_train_search = X.iloc[:train_size], y.iloc[:train_size]
    X_test_final, y_test_final = X.iloc[train_size:], y.iloc[train_size:]
    
    # Función objetivo para Optuna
    def objective(trial):
        # Espacio matemático de búsqueda para XGBoost
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        fold_maes = []
        
        for train_idx, val_idx in tscv.split(X_train_search):
            X_fold_train, X_fold_val = X_train_search.iloc[train_idx], X_train_search.iloc[val_idx]
            y_fold_train, y_fold_val = y_train_search.iloc[train_idx], y_train_search.iloc[val_idx]
            
            # Entrenamiento del modelo XGBoost
            model = xgb.XGBRegressor(**param)
            model.fit(X_fold_train, y_fold_train, 
                      eval_set=[(X_fold_val, y_fold_val)], 
                      verbose=False)
            
            preds = model.predict(X_fold_val)
            fold_maes.append(mean_absolute_error(y_fold_val, preds))
            
        return np.mean(fold_maes)

    # Ejecución del Estudio Optuna (20 iteraciones inteligentes)
    study = optuna.create_study(direction='minimize')
    print("Buscando el modelo óptimo (esto tomará unos minutos)...")
    study.optimize(objective, n_trials=20)
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"\nBúsqueda Bayesiana finalizada en {elapsed_time:.2f} minutos.")
    print("Mejores hiperparámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Entrenamiento y Evaluación Final con los mejores parámetros
    print("\nEvaluando agente XGBoost en el conjunto de prueba (Test Set)...")
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train_search, y_train_search)
    
    predicciones_finales = final_model.predict(X_test_final)
    mae_final = mean_absolute_error(y_test_final, predicciones_finales)
    
    print("-" * 60)
    print(f"MAE del Agente XGBoost (Test Set): {mae_final:.2f} MW")
    
    print("\nEntrenando agente maestro final con todo el dataset histórico...")
    final_model.fit(X, y)
    
    return final_model