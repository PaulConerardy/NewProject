import json
import pandas as pd
from datetime import datetime

def load_config(config_file='config.json'):
    """
    Charge les paramètres de configuration à partir d'un fichier JSON.

    Paramètres :
    - config_file (str) : Chemin vers le fichier de configuration.

    Retourne :
    - dict : Dictionnaire des paramètres de configuration.
    """
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier de configuration {config_file} n'a pas été trouvé.")
    except json.JSONDecodeError:
        raise ValueError(f"Erreur de décodage JSON dans le fichier {config_file}. Assurez-vous que le fichier est correctement formaté.")

def load_data(file_path):
    """
    Charge les données de transactions à partir d'un fichier CSV.

    Paramètres :
    - file_path (str) : Chemin vers le fichier CSV.

    Retourne :
    - pd.DataFrame : DataFrame contenant les données de transactions.
    """
    df = pd.read_csv(file_path, sep='|')
    df = df[df['TIMESTAMP'] != '2006-02-29']  # Suppression des dates invalides
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df['date'] = df['TIMESTAMP'].apply(lambda x: x.toordinal())  # Conversion en jours ordonnés
    return df
