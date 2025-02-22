import os
import pandas as pd
import pickle

def save_data(df, file_path, file_format="csv", overwrite=False):
    """
    Sauvegarde un DataFrame dans un format spécifique (CSV ou Pickle).
    Le param overwrite: permet d' écraser le fichier s'il existe déjà (False par défaut)
    """
    
    # Définir le chemin du fichier avec l'extension appropriée
    if file_format == "csv":
        file_path += ".csv"
    elif file_format == "pickle":
        file_path += ".pkl"
    else:
        raise ValueError("Format non pris en charge. Utiliser 'csv' ou 'pickle'.")
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(file_path) and not overwrite:
        print(f"Le fichier {file_path} existe déjà. Utilisez overwrite=True pour l'écraser.")
        return
    
    try:
        if file_format == "csv":
            df.to_csv(file_path, index=False)
        elif file_format == "pickle":
            with open(file_path, "wb") as f:
                pickle.dump(df, f)
        print(f"Données sauvegardées avec succès dans {file_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier : {e}")
