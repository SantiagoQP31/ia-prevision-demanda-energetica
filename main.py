from src.data_processing import load_and_preprocess_data
from src.model_training import train_and_evaluate

def main():
    filepath = 'data/energy_dataset.csv'

    print("--- Fase 1 y 2: Preprocesamiento y Feature Engineering ---")
    X, y, feature_name = load_and_preprocess_data(filepath)
    print(f"Dimensiones de la matriz de entrenamiento: {X.shape}")

    print("\n--- Fase 3 y 4: Entramiento y Evaluación ---")
    modelo_final = train_and_evaluate(X, y)

    print("\nProceso de entrenamiento completado. El agente base está listo para optimización.")

if __name__ == '__main__':
    main()