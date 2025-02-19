import pandas as pd
from sklearn.preprocessing import LabelEncoder

def transform_categorical(df: pd.DataFrame, method: str = 'onehot') -> pd.DataFrame:
    cat_columns = [col for col in df.columns
                   if df[col].dtype == 'object' or str(df[col].dtype) == 'category']

    if method.lower() == 'onehot':
        df_transformed = pd.get_dummies(df, columns=cat_columns)
    elif method.lower() == 'label':
        df_transformed = df.copy()
        for col in cat_columns:
            le = LabelEncoder()
            df_transformed[col] = le.fit_transform(df_transformed[col].astype(str))
    else:
        raise ValueError("Méthode non valide. Choisissez 'onehot' ou 'label'.")

    return df_transformed


if __name__ == "__main__":
    data = {
        'couleur': ['rouge', 'bleu', 'vert', 'rouge', 'bleu'],
        'forme': ['rond', 'carré', 'triangle', 'rond', 'triangle'],
        'valeur': [1, 2, 3, 1, 2]
    }
    df_example = pd.DataFrame(data)

    print("DataFrame original :")
    print(df_example)

    df_onehot = transform_categorical(df_example, method='onehot')
    print("\nDataFrame après One-Hot Encoding :")
    print(df_onehot)

    df_label = transform_categorical(df_example, method='label')
    print("\nDataFrame après Label Encoding :")
    print(df_label)
