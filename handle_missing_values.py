import pandas as pd



# SELECTIONNER LES COLONNES AVEC VALEURS MANQUANTES   
def fetch_nan_columns(data:pd.DataFrame) -> pd.DataFrame:
     
    """
     Selectionne et retourne  les colonnes  qui contiennent des valeurs manquantes 
     d'un dataframe
    """

    # Sélectionner les colonnes avec des valeurs manquantes
    colonnes_nan = data.columns[data.isnull().any()]

    # Filtrer le DataFrame pour ne garder que ces colonnes
    df_nan = data[colonnes_nan]

    return df_nan



def fetch_more_50_nan_columns(data:pd.DataFrame)->pd.DataFrame:

    """
     Selectionne et retourne  les colonnes  qui contiennent plus de 50% 
     de valeurs manquantes  d'un dataframe
    """

    # Sélectionner les colonnes avec des valeurs manquantes
    colonnes_nan = data.columns[data.isnull().any()]

    # Filtrer le DataFrame pour ne garder que les colonnes avec des nan
    df_nan = data[colonnes_nan]

    # Calcul du pourcentage de valeurs manquantes par colonne
    pourcentage_nan = df_nan.isnull().mean() * 100

    # Sélection des colonnes avec plus de 50% de valeurs manquantes
    colonnes_a_supprimer = pourcentage_nan[pourcentage_nan > 50].index

    # Filtrer le DataFrame
    df_filtre = df_nan[colonnes_a_supprimer]

    return df_filtre



# AFFICHER LE POURCENTAGE DE VALEURS MANQUANTES DANS UNE COLONNE SPECIFIQUE

def nan_column_percent(data:pd.DataFrame,colname:str):
    """
    Affiche  le pourcentage de valeurs manquantes  
    d'une colonne spécifique  d'un dataframe
    """
    # Sélectionner les colonnes avec des valeurs manquantes
    colonnes_nan = data.columns[data.isnull().any()]

    # Filtrer le DataFrame pour ne garder que ces colonnes
    df_nan = data[colonnes_nan]

    if colname in df_nan.columns:

        # Calcul du pourcentage de valeurs manquantes sur la colonne spécifique
        pourcentage_nan = df_nan[colname].isnull().mean() * 100

        print(f"Pourcentage de valeurs manquantes dans la colonne '{colname}': {pourcentage_nan:.2f}%")

    else: 
        print("la  colonne spécifiée n'appartient pas à l'ensemble des colonnes contenant des valeurs manquantes")



# IMPUTATIONS DES VALEURS MANQUANTES 


# Variables quantitatives 

def impute_nan_with_mean(data:pd.DataFrame) -> None:
    """ 
    Impute les variables quantitatives manquantes  avec le la moyenne 
    """
    for col in data.select_dtypes('number').columns:
        if -0.5 < data[col].skew() < 0.5:
            data[col] = data[col].fillna(data[col].mean())



def impute_nan_with_median(data:pd.DataFrame) -> None:

    """ 
    Impute les variables quantitatives manquantes  avec le la mediane 
    """
    for col in data.select_dtypes('number').columns:
        if not (-0.5 < data[col].skew() < 0.5):
            data[col] = data[col].fillna(data[col].median())



# Variables qualitatives 

def impute_nan_with_mode(data:pd.DataFrame) -> None:
    """ Impute les variables qualitatives manquantes  avec le mode """
    for col in data.select_dtypes('object').columns:
        data[col] = data[col].fillna(data[col].mode()[0])


# SUPPRIMER LES LIGNES OU COLONNES CONTENANT DES  VALEURS MANQUANTES  


# les lignes contenant au moins une valeur manquantes

def drop_nan_rows (data:pd.DataFrame) -> pd.DataFrame:
    """
     Supprime les lignes contenant des valeurs manquantes
    """
    df=data.dropna()
    return df

# les colonnes avec  plus de 50%  de valeurs manquantes 


def drop_more_50_nan_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes qui contiennent plus de 50% de valeurs manquantes dans un DataFrame.
    
    :param data: DataFrame Pandas d'entrée
    :return: DataFrame sans les colonnes ayant plus de 50% de NaN
    """
    # Calcul du pourcentage de valeurs manquantes par colonne
    pourcentage_nan = data.isnull().mean() * 100

    # Sélection des colonnes à supprimer (plus de 50% de NaN)
    colonnes_a_supprimer = pourcentage_nan[pourcentage_nan > 50].index

    # Supprimer ces colonnes du DataFrame
    data_cleaned = data.drop(columns=colonnes_a_supprimer)

    return data_cleaned
