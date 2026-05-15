import pandas as pd

def load_and_preprocess_data(energy_path, weather_path):

    # Carga e indexacion temporal
    df_energy = pd.read_csv(energy_path)
    df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)
    df_energy.set_index('time', inplace=True)

    target_col = 'total load actual'
    df_energy = df_energy.dropna(subset=[target_col]).copy()

    # Carga y agregacion del clima (Nuevos sensores)
    print("Procesando y agregando datos meteorologicos...")
    df_weather = pd.read_csv(weather_path)
    df_weather['dt_iso'] = pd.to_datetime(df_weather['dt_iso'], utc=True)

    # Agrupamos por hora y sacamos el promedio nacional de las metricas climaticas clave
    weather_agg = df_weather.groupby('dt_iso')[['temp', 'humidity', 'wind_speed']].mean()
    weather_agg.index.name = 'time' # Renombramos el índice para que coincida

    df = df_energy.join(weather_agg, how='inner')

    # feature engineering (Calendario, lags, medias moviles)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x > 5 else 0)

    df['lag_1'] = df[target_col].shift(1)
    df['lag_24'] = df[target_col].shift(24)
    df['rolling_mean_24'] = df[target_col].rolling(window=24).mean()

    # Limpieza de nulos generados por rezagos
    df.dropna(subset=['lag_1', 'lag_24', 'rolling_mean_24'], inplace=True)

    # Definicion de matriz X y vector y
    features = ['lag_1', 'lag_24', 'rolling_mean_24', 'hour', 'dayofweek', 'month', 'is_weekend', 'temp', 'humidity', 'wind_speed']
    
    X = df[features]
    y = df[target_col]

    return X, y, features
