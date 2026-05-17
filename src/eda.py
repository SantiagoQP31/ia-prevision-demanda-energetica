import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

def run_eda(energy_path, weather_path):
    os.makedirs('plots', exist_ok=True)
    print("=" * 70)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 70)

    # Carga
    df_energy = pd.read_csv(energy_path)
    df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)
    df_energy.set_index('time', inplace=True)

    target_col = 'total load actual'

    # 1. CALIDAD DE DATOS
    print("\n1. CALIDAD DE DATOS")
    print("-" * 70)
    print(f"Filas totales: {len(df_energy)}")
    print(f"Missing values en '{target_col}': {df_energy[target_col].isna().sum()}")
    print(f"Período temporal: {df_energy.index.min()} a {df_energy.index.max()}")

    # 2. ESTADÍSTICAS DEL TARGET
    print("\n2. ESTADÍSTICAS DEL TARGET (Consumo en MW)")
    print("-" * 70)
    stats = df_energy[target_col].describe()
    print(stats)
    print(f"Asimetría (skewness): {df_energy[target_col].skew():.4f}")
    print(f"Curtosis: {df_energy[target_col].kurtosis():.4f}")

    # 3. DETECCIÓN DE OUTLIERS
    print("\n3. DETECCIÓN DE OUTLIERS")
    print("-" * 70)
    Q1 = df_energy[target_col].quantile(0.25)
    Q3 = df_energy[target_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_energy[(df_energy[target_col] < Q1 - 1.5*IQR) | (df_energy[target_col] > Q3 + 1.5*IQR)]
    print(f"Outliers detectados (IQR): {len(outliers)} ({100*len(outliers)/len(df_energy):.2f}%)")
    if len(outliers) > 0:
        print(f"  Min outlier: {outliers[target_col].min():.2f} MW")
        print(f"  Max outlier: {outliers[target_col].max():.2f} MW")

    # 4. VISUALIZACIONES
    print("\n4. GENERANDO VISUALIZACIONES...")
    print("-" * 70)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 4.1 Serie temporal (últimos 30 días)
    ax = axes[0, 0]
    data_30d = df_energy[target_col].iloc[-30*24:]
    ax.plot(data_30d, linewidth=1, color='#2c3e50')
    ax.set_title('Consumo últimos 30 días', fontweight='bold')
    ax.set_ylabel('MW')
    ax.grid(True, alpha=0.3)

    # 4.2 Distribución
    ax = axes[0, 1]
    ax.hist(df_energy[target_col], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_title('Distribución del consumo', fontweight='bold')
    ax.set_xlabel('MW')
    ax.set_ylabel('Frecuencia')
    ax.grid(True, alpha=0.3, axis='y')

    # 4.3 Box plot por hora
    ax = axes[1, 0]
    df_energy['hour'] = df_energy.index.hour
    df_energy.boxplot(column=target_col, by='hour', ax=ax)
    ax.set_title('Consumo por hora del día', fontweight='bold')
    ax.set_xlabel('Hora')
    ax.set_ylabel('MW')
    plt.sca(ax)
    plt.xticks(range(1, 25, 2), range(0, 24, 2))

    # 4.4 Box plot por día de semana
    ax = axes[1, 1]
    df_energy['dayofweek'] = df_energy.index.dayofweek
    days_name = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    df_energy.boxplot(column=target_col, by='dayofweek', ax=ax)
    ax.set_title('Consumo por día de semana', fontweight='bold')
    ax.set_xlabel('Día')
    ax.set_ylabel('MW')
    plt.sca(ax)
    plt.xticks(range(1, 8), days_name)

    # 4.5 Autocorrelación (lag_1)
    ax = axes[2, 0]
    lag_1_corr = df_energy[target_col].corr(df_energy[target_col].shift(1))
    lag_24_corr = df_energy[target_col].corr(df_energy[target_col].shift(24))
    lags = [1, 2, 3, 6, 12, 24, 48, 72]
    corrs = [df_energy[target_col].corr(df_energy[target_col].shift(l)) for l in lags]
    ax.bar(range(len(lags)), corrs, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([f'lag{l}' for l in lags], rotation=45)
    ax.set_title('Autocorrelación por rezago', fontweight='bold')
    ax.set_ylabel('Correlación con consumo actual')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 4.6 Descomposición STL
    ax = axes[2, 1]
    ax.axis('off')
    info_text = f"""
    HALLAZGOS CLAVE:

    • Correlación lag_1: {lag_1_corr:.4f}
    • Correlación lag_24: {lag_24_corr:.4f}

    • Media: {df_energy[target_col].mean():.2f} MW
    • Desv. Estándar: {df_energy[target_col].std():.2f} MW
    • Min: {df_energy[target_col].min():.2f} MW
    • Max: {df_energy[target_col].max():.2f} MW

    • Rango de variación: {df_energy[target_col].max() - df_energy[target_col].min():.2f} MW
    """
    ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('plots/eda_analisis.png', dpi=300, bbox_inches='tight')
    print("✓ Gráficos guardados como 'plots/eda_analisis.png'")

    # 5. DESCOMPOSICIÓN STL
    print("\n5. DESCOMPOSICIÓN DE LA SERIE TEMPORAL")
    print("-" * 70)
    try:
        # Usar últimos 60 días para descomposición (más rápido)
        data_for_decomp = df_energy[target_col].iloc[-60*24:]
        decomposition = seasonal_decompose(data_for_decomp, model='additive', period=24)

        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        decomposition.observed.plot(ax=axes[0], color='#2c3e50')
        axes[0].set_ylabel('Observado')
        axes[0].set_title('Descomposición STL (últimos 60 días)', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        decomposition.trend.plot(ax=axes[1], color='#3498db')
        axes[1].set_ylabel('Tendencia')
        axes[1].grid(True, alpha=0.3)

        decomposition.seasonal.plot(ax=axes[2], color='#2ecc71')
        axes[2].set_ylabel('Estacionalidad')
        axes[2].grid(True, alpha=0.3)

        decomposition.resid.plot(ax=axes[3], color='#e74c3c')
        axes[3].set_ylabel('Residual')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('plots/eda_descomposicion.png', dpi=300, bbox_inches='tight')
        print("✓ Descomposición guardada como 'plots/eda_descomposicion.png'")
        print(f"  Amplitud de estacionalidad diaria: {decomposition.seasonal.max() - decomposition.seasonal.min():.2f} MW")
    except Exception as e:
        print(f"⚠ Error en descomposición: {str(e)}")

    # 6. ANÁLISIS DE DATOS METEOROLÓGICOS
    print("\n6. ANÁLISIS DE DATOS METEOROLÓGICOS")
    print("-" * 70)
    df_weather = pd.read_csv(weather_path)
    df_weather['dt_iso'] = pd.to_datetime(df_weather['dt_iso'], utc=True)
    weather_agg = df_weather.groupby('dt_iso')[['temp', 'humidity', 'wind_speed']].mean()
    print(f"Temperatura - Min: {weather_agg['temp'].min():.2f}°C, Max: {weather_agg['temp'].max():.2f}°C, Media: {weather_agg['temp'].mean():.2f}°C")
    print(f"Humedad - Min: {weather_agg['humidity'].min():.2f}%, Max: {weather_agg['humidity'].max():.2f}%, Media: {weather_agg['humidity'].mean():.2f}%")
    print(f"Viento - Min: {weather_agg['wind_speed'].min():.2f} m/s, Max: {weather_agg['wind_speed'].max():.2f} m/s, Media: {weather_agg['wind_speed'].mean():.2f} m/s")

    print("\n" + "=" * 70)
    print("EDA COMPLETADO")
    print("=" * 70)

    return df_energy

if __name__ == '__main__':
    energy_path = 'data/energy_dataset.csv'
    weather_path = 'data/weather_features.csv'
    run_eda(energy_path, weather_path)
