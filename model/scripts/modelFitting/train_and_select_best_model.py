
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def save_data(model, filename):
    joblib.dump(model, filename)
    print("Modèle sauvegardé dans", filename)


def train_and_select_model(data_frame, preproc_pipeline, target, models_dict,
                           test_size=0.2, cv=3, scoring='accuracy', random_state=42):
    X = data_frame.drop(columns=[target])
    y = data_frame[target]

    # Découpage en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)

    results = {}
    best_score = -np.inf
    best_model = None

    for model_name, config in models_dict.items():
        print(f"Entraînement du modèle: {model_name}")
        model_instance = config['model']
        param_grid = config['params']

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preproc_pipeline),
            ('model', model_instance)
        ])

        grid_search = GridSearchCV(model_pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        roc_auc = None
        if len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])

        results[model_name] = {
            'best_cv_score': grid_search.best_score_,
            'test_accuracy': test_acc,
            'roc_auc': roc_auc,
            'best_params': grid_search.best_params_
        }

        print("Modèle:", model_name)
        print("  Score CV:", grid_search.best_score_)
        print("  Accuracy sur test:", test_acc)
        if roc_auc is not None:
            print("  ROC AUC sur test:", roc_auc)
        print("  Meilleurs paramètres:", grid_search.best_params_)
        print("------")

        if test_acc > best_score:
            best_score = test_acc
            best_model = grid_search.best_estimator_

    model_names = list(results.keys())
    scores = [results[m]['test_accuracy'] for m in model_names]

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, scores, color='skyblue')
    plt.xlabel("Modèles")
    plt.ylabel("Accuracy sur test")
    plt.title("Comparaison des performances des modèles")
    plt.ylim(0, 1)
    plt.show()

    save_data(best_model, "best_model.pkl")

    return best_model, results


if __name__ == "__main__":
    data = pd.DataFrame({
        'longitude': [-122.23, -122.22, -122.24, -122.25, -122.26, -122.27, -122.28, -122.29, -122.30, -122.31],
        'latitude': [37.88, 37.86, 37.85, 37.84, 37.83, 37.82, 37.81, 37.80, 37.79, 37.78],
        'housing_median_age': [41, 21, 52, 52, 30, 25, 40, 35, 45, 50],
        'total_rooms': [880, 7099, 1467, 1274, 1500, 1600, 1700, 1800, 1900, 2000],
        'total_bedrooms': [129, 1106, 190, 235, 210, 220, 230, 240, 250, 260],
        'population': [322, 2401, 496, 558, 600, 700, 800, 900, 1000, 1100],
        'households': [126, 1138, 177, 219, 200, 210, 220, 230, 240, 250],
        'median_income': [8.3252, 8.3014, 7.2574, 5.6431, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
        'median_house_value': [452600, 358500, 352100, 341300, 360000, 370000, 380000, 390000, 400000, 410000],
        'cat_feature': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'B', 'A'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                          'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']

    data['cat_feature'] = data['cat_feature'].astype('category')
    categorical_features = ['cat_feature']

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preproc_pipeline = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])

    models_dict = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {'model__C': [0.1, 1, 10]}
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'params': {'model__n_estimators': [50, 100],
                       'model__max_depth': [None, 5, 10]}
        },
        'SVC': {
            'model': SVC(probability=True),
            'params': {'model__C': [0.1, 1, 10],
                       'model__gamma': ['scale', 'auto']}
        },
        'XGBClassifier': {
            'model': XGBClassifier(),
            'params': {'model__n_estimators': [50, 100],
                       'model__max_depth': [3, 5, 7]}
        }
    }

    best_model, results = train_and_select_model(data, preproc_pipeline, target='target',
                                                 models_dict=models_dict)
    print("Meilleur modèle:", best_model)
    print("Résultats:", results)
