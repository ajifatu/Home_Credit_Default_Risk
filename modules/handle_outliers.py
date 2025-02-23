import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from modules import handle_outliers as ho


# TRAITER LES VALEURS ABERRANTES 

# pour les données(distribution) asymétriques 
def replace_outliers_IQR(data,factor=1.5)->pd.DataFrame:
    data=data.copy()
    for col in data.select_dtypes('number').columns:
        if not (-0.5 <= data[col].skew() <= 0.5):
            Q1 = np.quantile(data[col], 0.25)
            Q3 = np.quantile(data[col], 0.75)
            IQR = Q3 - Q1
            limitInf = Q1 - factor*IQR  
            limitSup = Q3 + factor*IQR

            data[col] = np.where(data[col] <= limitInf, limitInf, data[col])
            data[col] = np.where(data[col] >= limitSup, limitSup, data[col])
    return data


# pour les données symétriques 

def replace_outliers_Zscore(data:pd.DataFrame, threshold:int=3) -> pd.DataFrame:
    data=data.copy()
    for col in data.select_dtypes('number').columns:
        if -0.5 <= data[col].skew() <= 0.5:
            u = np.mean(data[col])   
            sigma = np.std(data[col]) 
            median = np.median(data[col]) 
            z =  (data[col] - u)/sigma
            data[col] = np.where(abs(z) > threshold, median, data[col])
    return data    



# méthode  écrite par Seydou
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
            # attention ici
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        elif method == 'zscore':
            # attention ici 
            z_scores = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
            df_clean = df_clean[np.abs(z_scores) <= z_thresh]

    
    return df_clean



# pour les données asymétriques 
def handle_outlier_winsor(data:pd.DataFrame) -> pd.DataFrame:
    data=data.copy()
    ''' ca permet de remplacer les valeurs aberrantes par des percentiles calcules à partir seuils choisis'''
    for col in data.select_dtypes('number').columns:
        if not (-0.5 <= data[col].skew() <=0.5):
            percentileInf = np.percentile(data[col], 7)
            percentileSup = np.percentile(data[col], 93)
            data[col] = np.clip(data[col], percentileInf, percentileSup)
    return data  



def handle_outliers(data:pd.DataFrame) -> pd.DataFrame:
    data=data.copy()
    # Sélection des colonnes numériques
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    # Identification des colonnes dichotomiques (0 et 1)
    cols_dichotomiques = [col for col in numeric_cols if data[col].dropna().isin([0, 1]).all()]
    
    # Colonnes à exclure du scaling
    cols_exclues = ["SK_ID_CURR", "SK_BUREAU_ID", "SK_ID_PREV"]
    
    # Colonnes à imputer les outliers
    cols_to_impute_outliers = [col for col in numeric_cols if col not in cols_dichotomiques + cols_exclues]
    for col in cols_to_impute_outliers:
        if data[col].skew() < -0.5 or data[col].skew() > 0.5:
            data[col] = replace_outliers_IQR(data[[col]])
        else:
            data[col] = replace_outliers_Zscore(data[[col]])
    
    return data