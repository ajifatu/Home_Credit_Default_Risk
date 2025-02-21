import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler ,StandardScaler



# AFFICHER LES DISTRIBUTIONS (HISTOGRAMMES  ET BOXPLOTS)

def distributions(df, nrows, ncols):
  """
  Affiche histogrammes et boxplotS  des colonnes numériques d'un dataframe
  """
  for col in df.select_dtypes('number').columns:
    fig, axes = plt.subplots(nrows, ncols)
    sns.histplot(df[col], bins= 20,ax = axes[0])
    sns.boxplot(x = df[col], ax = axes[1], showmeans=True)
  plt.show()


# TRAITER LES VALEURS ABERRANTES 

# pour les données asymétriques 
def replace_outliers_IQR(data,factor=1.5)->pd.DataFrame:
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
    ''' ca permet de remplacer les valeurs aberrantes par des percentiles calcules à partir seuils choisis'''
    for col in data.select_dtypes('number').columns:
        if not (-0.5 <= data[col].skew() <=0.5):
            percentileInf = np.percentile(data[col], 7)
            percentileSup = np.percentile(data[col], 93)
            data[col] = np.clip(data[col], percentileInf, percentileSup)
    return data  



#  FEATURE SCALING 


   


def scaling_2(data:pd.DataFrame) -> pd.DataFrame:
     
    """
     Cette fonction permet de mettre les colonnes numériques d'un dataset 
     sur une meme echelle, elle suppose qu'il n'y a pas de NaN
    """

    # selectionnes  uniquement les colonnes numériques sauf  des identifiants(SK_ID_CURR par eg)
    for col in data.select_dtypes('number').columns:

        if col not in ["SK_ID_CURR","SK_BUREAU_ID","SK_ID_PREV"]:
                # s'assurer que les valeurs aberrantes et NaN sont traitées aussi bien pour les colonnes symétriques qu'asymétriques
                
                if data[col].isna().sum().sum()== 0:
                        if not (-0.5 <= data[col].skew() <=0.5):
                            # remplacer les valeurs aberrantes par Q1-1,5IQR ou  Q3-1,5IQR 
                            data[col]=replace_outliers_IQR(data[col])
                            # mettre les données sur la  meme echelle  avec MinMaxScaler
                            data[col]=pd.DataFrame(MinMaxScaler().fit_transform(data[[col]]),columns=data[[col]])
                            return data
                        else: 
                            data[col]=replace_outliers_Zscore(data[col])
                            data[col]=pd.DataFrame(StandardScaler().fit_transform(data[[col]]),columns=data[[col]])
                else:
                    col_na_number=data[col].isna().sum().sum()
                    print(f"La colonne{col} contient {col_na_number} valeurs manquantes ")
    return data   



def scaling_with_na(data: pd.DataFrame) -> pd.DataFrame:
    """
    Met toutes les colonnes numériques d'un dataset sur une même échelle,
    tout en gérant les valeurs aberrantes et les NaN.
    """

    data = data.copy()  # Éviter de modifier le dataset original
    cols_to_scale = [col for col in data.select_dtypes('number').columns if col not in ["SK_ID_CURR", "SK_BUREAU_ID", "SK_ID_PREV"]]
    
    # Création des scalers
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    
    for col in cols_to_scale:
        # Remplacement des NaN par la médiane
        data[col].fillna(data[col].median(), inplace=True)
        
        # Vérification de l'asymétrie
        if not (-0.5 <= data[col].skew() <= 0.5):  # Asymétrique
            data[col] = replace_outliers_IQR(data[col])
            data[col] = minmax_scaler.fit_transform(data[[col]])  # MinMaxScaler
        else:  # Symétrique
            data[col] = replace_outliers_Zscore(data[col])
            data[col] = standard_scaler.fit_transform(data[[col]])  # StandardScaler
            
    return data
