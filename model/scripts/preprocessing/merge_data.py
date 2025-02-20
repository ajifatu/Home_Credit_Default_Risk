import pandas as pd
import numpy as np

def reduce_memory_usage(df):
    """Optimise l'utilisation de la mémoire en convertissant les types de données."""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:  # Exclure les chaînes de caractères
            min_val, max_val = df[col].min(), df[col].max()
            
            if col_type == 'int64':
                if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            elif col_type == 'float64':
                if min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    return df

def merge_tables_on_key(tables, key_column, how="left"):
    """
    Fusionne plusieurs DataFrames sur une colonne clé commune après filtrage des tables ne contenant pas la clé.
    Optimise aussi la mémoire avant la fusion.
    """
    if not tables:
        raise ValueError("La liste de tables est vide.")

    # Filtrer les tables contenant bien la colonne clé
    valid_tables = [df for df in tables if key_column in df.columns]

    if not valid_tables:
        raise KeyError(f"Aucune table ne contient la colonne clé '{key_column}'.")

    # Afficher un avertissement si certaines tables sont ignorées
    ignored_tables = len(tables) - len(valid_tables)
    if ignored_tables > 0:
        print(f"Avertissement : {ignored_tables} table(s) ignorée(s) car elles ne contiennent pas '{key_column}'.")

    # Réduction de la mémoire et harmonisation du type de clé
    for i, df in enumerate(valid_tables):
        df = reduce_memory_usage(df)  # Optimiser la mémoire
        df[key_column] = df[key_column].astype(str)  # Harmoniser les clés
        valid_tables[i] = df  # Mettre à jour la liste

    # Fusion progressive des DataFrames valides
    merged_df = valid_tables[0]
    for df in valid_tables[1:]:
        merged_df = merged_df.merge(df, on=key_column, how=how)

    return merged_df

# Exemple d'utilisation
data1 = pd.read_csv("bureau.csv")
data2 = pd.read_csv("bureau_balance.csv")
data3 = pd.read_csv("application_train.csv")

tables = [data1, data2, data3]

try:
    merged_result = merge_tables_on_key(tables, key_column="SK_ID_BUREAU", how="left")
    print(merged_result.info())  # Afficher la structure après fusion
except KeyError as e:
    print(f"Erreur : {e}")
except MemoryError:
    print("Erreur : Problème de mémoire. Essayez d'augmenter la RAM ou d'utiliser des solutions comme Dask.")
