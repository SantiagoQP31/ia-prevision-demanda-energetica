from src.data_processing import load_and_preprocess_data
from src.model_training import train_and_evaluate

def main():
    energy_path = 'data/energy_dataset.csv'
    weather_path = 'data/weather_features.csv'

    print("--- Fase 1 y 2: Preprocesamiento y Feature Engineering de Características Multivariadas ---")
    X, y, feature_names = load_and_preprocess_data(energy_path, weather_path)
    print(f"Dimensiones de la matriz de entrenamiento: {X.shape}")

    print(f"Dimensiones de la nueva matriz de entrenamiento: {X.shape}")
    print(f"Sensores activos: {feature_names}")

    print("\n--- Fase 3 y 4: Entrenamiento y Evaluación ---")
    modelo_final = train_and_evaluate(X, y)

    print("\nProceso de entrenamiento completado. El agente base está listo para optimización.")

if __name__ == '__main__':
    main()