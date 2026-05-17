import numpy as np
import pandas as pd
import holidays

def load_and_preprocess_data(energy_path, weather_path):
    # Carga e Indexación de Energía
    df_energy = pd.read_csv(energy_path)
    df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)
    df_energy.set_index('time', inplace=True)
    
    target_col = 'total load actual'
    df_energy = df_energy.dropna(subset=[target_col]).copy()
    
    # Carga y Agregación del Clima
    print("Procesando y agregando datos meteorológicos...")
    df_weather = pd.read_csv(weather_path)
    df_weather['dt_iso'] = pd.to_datetime(df_weather['dt_iso'], utc=True)
    weather_agg = df_weather.groupby('dt_iso')[['temp', 'humidity', 'wind_speed']].mean()
    weather_agg.index.name = 'time'
    
    # Fusión de Entornos
    df = df_energy.join(weather_agg, how='inner')
    
    # Inyección del Calendario de Festivos (Nuevo Punto Ciego Cubierto)
    print("Inyectando calendario de festividades de España...")
    anios_dataset = df.index.year.unique().tolist()
    festivos_es = holidays.Spain(years=anios_dataset)
    # Comparamos si la fecha de cada fila coincide con un día festivo
    df['is_holiday'] = df.index.map(lambda x: 1 if x.date() in festivos_es else 0)
    
    # Ingeniería de Características Avanzada 
    
    # Variables base de calendario
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # Rezagos de Memoria a Largo Plazo
    df['lag_24'] = df[target_col].shift(24)
    df['lag_168'] = df[target_col].shift(168)  # 168 = 7 días * 24 horas
    df['rolling_mean_24'] = df[target_col].rolling(window=24).mean()
    
    df.dropna(subset=['lag_24', 'lag_168', 'rolling_mean_24'], inplace=True)
    
    # Definición de la matriz X final
    features = ['lag_24', 'lag_168', 'rolling_mean_24', 
                'hour', 'dayofweek', 'month', 'is_weekend', 'is_holiday', 
                'temp', 'humidity', 'wind_speed']
    
    X = df[features]
    y = df[target_col]
    
    return X, y, features