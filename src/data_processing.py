import pandas as pd

def load_and_preprocess_data(filepath):

    # Carga e indexacion temporal
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)

    target_col = 'total load actual'
    df = df.dropna(subset=[target_col]).copy()

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
    features = ['lag_1', 'lag_24', 'rolling_mean_24', 'hour', 'dayofweek', 'month', 'is_weekend']
    X = df[features]
    y = df[target_col]

    return X, y, features
