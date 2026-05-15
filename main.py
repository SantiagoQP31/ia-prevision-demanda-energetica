import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing import load_and_preprocess_data
from src.model_training import train_and_evaluate

def extract_feature_importance(model, feature_names):
    print("\n--- Extracción de Conocimiento del Agente ---")
    importancias = model.feature_importances_
    df_importancias = pd.DataFrame({'Variable': feature_names, 'Importancia': importancias})
    df_importancias = df_importancias.sort_values(by='Importancia', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(df_importancias['Variable'], df_importancias['Importancia'], color='#2c3e50')
    plt.xlabel('Importancia Relativa (Impureza de Gini / MSE)')
    plt.title('Ponderación de Sensores en la Toma de Decisiones del Agente')
    plt.tight_layout()
    plt.savefig('feature_importance.png', format='png', dpi=300)
    print("Gráfico de importancia exportado como 'feature_importance.png'")

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

    extract_feature_importance(modelo_final, feature_names)

    print("\nProceso de entrenamiento completado. El agente base está listo para optimización.")

if __name__ == '__main__':
    main()