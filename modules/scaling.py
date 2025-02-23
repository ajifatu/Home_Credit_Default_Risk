
import pandas as pd
from modules import handle_outliers as hs 
from sklearn.preprocessing import MinMaxScaler ,StandardScaler,RobustScaler


#  FEATURE SCALING 


def scaling(data: pd.DataFrame) -> pd.DataFrame:
    """
    Met toutes les colonnes numériques d'un dataset sur une même échelle,
    en excluant les variables dichotomiques (0 et 1).

    Cette fonction suppose que les valeurs manquantes et aberrantes sont traités
    """
    data = data.copy()
    
    # Sélection des colonnes numériques
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    # Identification des colonnes dichotomiques (0 et 1)
    cols_dichotomiques = [col for col in numeric_cols if data[col].dropna().isin([0, 1]).all()]
    
    # Colonnes à exclure du scaling
    cols_exclues = ["SK_ID_CURR", "SK_BUREAU_ID", "SK_ID_PREV"]
    
    # Colonnes à scaler
    cols_to_scale = [col for col in numeric_cols if col not in cols_dichotomiques + cols_exclues]
    
    # Initialisation des scalers
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    
    # Application du scaling
    for col in cols_to_scale:
        if data[col].skew() < -0.5 or data[col].skew() > 0.5:
            data[col] = pd.Series(minmax_scaler.fit_transform(data[[col]]))
        else:
            data[col] = pd.Series(standard_scaler.fit_transform(data[[col]]))
    
    return data
