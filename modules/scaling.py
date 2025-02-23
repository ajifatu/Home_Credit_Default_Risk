
import pandas as pd
from modules import handle_outliers as hs 
from sklearn.preprocessing import MinMaxScaler ,StandardScaler,RobustScaler


#  FEATURE SCALING 

def scaling_robust_scaler(data: pd.DataFrame) -> pd.DataFrame:
    """
    Met toutes les colonnes numériques d'un dataset sur une même échelle.
    Applicable sur n'importe qu'elle  distribution sauf Normal 
    """

    data = data.copy()  # Éviter de modifier le dataset original
    
    cols_to_scale = [col for col in data.select_dtypes('number').columns if col not in ["SK_ID_CURR","TARGET", "SK_BUREAU_ID", "SK_ID_PREV"]]
    
    # Création des scalers
    robust_scaler=RobustScaler()
    
    data= pd.DataFrame(robust_scaler.fit_transform(data[cols_to_scale]), columns=cols_to_scale)

       
    return data


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
                            data[col]=hs.replace_outliers_IQR(data[col])
                            # mettre les données sur la  meme echelle  avec MinMaxScaler
                            data[col]=pd.DataFrame(MinMaxScaler().fit_transform(data[[col]]),columns=data[[col]])
                            return data
                        else: 
                            data[col]=hs.replace_outliers_Zscore(data[col])
                            data[col]=pd.DataFrame(StandardScaler().fit_transform(data[[col]]),columns=data[[col]])
                else:
                    col_na_number=data[col].isna().sum().sum()
                    print(f"La colonne{col} contient {col_na_number} valeurs manquantes ")
    return data   



def scaling(data: pd.DataFrame) -> pd.DataFrame:
    """
    Met toutes les colonnes numériques d'un dataset sur une même échelle,
    """

    data = data.copy()  # Éviter de modifier le dataset original
    cols_dichotomiques=[col for col in  data.select_dtypes("number").columns 
                    if data[col].isin([0,1]).all()]
    cols_to_scale = [col for col in data.select_dtypes('number').columns if col not in ["SK_ID_CURR","TARGET" "SK_BUREAU_ID", "SK_ID_PREV"]
                     and  col not in cols_dichotomiques]
    
    # Création des scalers
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    
    for col in cols_to_scale:
        
        # Vérification de l'asymétrie
        if not (-0.5 <= data[col].skew() <= 0.5):  # Asymétrique
            #data[col] = hs.replace_outliers_IQR(data[col])
            data[col] = pd.DataFrame(minmax_scaler.fit_transform(data[[col]]))  # MinMaxScaler
        else:  # Symétrique
            #data[col] = hs.replace_outliers_Zscore(data[col])
            data[col] = pd.DataFrame(standard_scaler.fit_transform(data[[col]]))  # StandardScaler
            
    return data