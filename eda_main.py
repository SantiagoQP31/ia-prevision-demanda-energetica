from src.eda import run_eda

if __name__ == '__main__':
    energy_path = 'data/energy_dataset.csv'
    weather_path = 'data/weather_features.csv'
    run_eda(energy_path, weather_path)
