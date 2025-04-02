from kaggle.api.kaggle_api_extended import KaggleApi

# Initialiser Kaggle API
api = KaggleApi()
api.authenticate()

# Last ned Crop Recommendation Dataset og pakk det ut til 'data/'-mappen i prosjektet
api.dataset_download_files('atharvaingle/crop-recommendation-dataset', path='C:/Users/matsw/OneDrive/Skrivebord/Intelligente Systemer/Assignment-E/data', unzip=True)

print("Dataset nedlastet og pakket ut til data-mappen!")
