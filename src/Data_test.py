import pandas as pd

# Last inn crop recommendation dataset fra data-mappen
data_path = 'C:/Users/matsw/OneDrive/Skrivebord/Intelligente Systemer/Assignment-E/data/Crop_recommendation.csv'
data = pd.read_csv(data_path)

# Sjekk de første radene av dataset
print(data.head())
print(data.isnull().sum())
print(data.describe())
print(data.dtypes)
