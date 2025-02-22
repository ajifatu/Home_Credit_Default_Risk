import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV en DataFrame.
    """
    
    # Vérifier si le fichier existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier '{file_path}' est introuvable.")
    
    try:
        # Charger les données
        df = pd.read_csv(file_path)
        
        # Afficher un aperçu des données
        print(f" Chargement réussi : {file_path}")
        print(f" Dimensions du DataFrame : {df.shape}")
        
        return df
    
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier : {e}")
