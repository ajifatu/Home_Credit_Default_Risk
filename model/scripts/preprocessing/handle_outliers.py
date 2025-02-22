import pandas as pd
import numpy as np

def remove_outliers(df, method='iqr', threshold=1.5, z_thresh=3):
    """
    Supprime les valeurs aberrantes d'un DataFrame en utilisant IQR ou Z-score.
    
    Paramètres:
    df : pd.DataFrame
        DataFrame d'entrée contenant des colonnes numériques.
    method : str, optional
        Méthode pour détecter les outliers ('iqr' pour intervalle interquartile, 'zscore' pour Z-score).
    threshold : float, optional
        Facteur de seuil pour la méthode IQR (par défaut 1.5).
    z_thresh : float, optional
        Seuil du Z-score pour la méthode Z-score (par défaut 3).
    Retourne:
    pd.DataFrame
        DataFrame nettoyé sans valeurs aberrantes.
    """
    df_clean = df.copy()
    
    for col in df_clean.select_dtypes(include=[np.number]):  # Sélectionner uniquement les colonnes numériques
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        elif method == 'zscore':
            z_scores = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
            df_clean = df_clean[np.abs(z_scores) <= z_thresh]
    
    return df_clean

# Exemple d'utilisation
df = pd.DataFrame({'A': [1, 2, 3, 100, 5], 'B': [10, 20, 30, 40, 500]})
df_cleaned = remove_outliers(df, method='iqr')
print(df_cleaned)
