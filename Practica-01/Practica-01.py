import os, zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

if os.path.exists("vehicles.csv"):
    print("El dataset fue creado correctamente")
else:
    print("Generando dataset")
    api.dataset_download_file('austinreese/craigslist-carstrucks-data','vehicles.csv')
    with zipfile.ZipFile("vehicles.csv.zip", "r") as f:
        f.extractall()
