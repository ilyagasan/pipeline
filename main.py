from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer

import pandas as pd
import numpy as np
import joblib
def filter_data(df):
   columns_to_drop = [
       'id',
       'url',
       'region',
       'region_url',
       'price',
       'manufacturer',
       'image_url',
       'description',
       'posting_date',
       'lat',
       'long'
   ]
   df1 = df.drop(columns_to_drop)
   return df1


def main():
    df = pd.read_csv('homework.csv')

    # x = df.drop('Loan_Status', axis=1)
    # y = df['Loan_Status'].apply(lambda x: 1.0 if x == 'Y' else 0.0)
    x = df.drop(['url', 'id', 'price_category', 'region_url', 'price', 'image_url', 'description', 'posting_date'], axis=1)
    y = df['price_category'].apply(lambda x: 1.0 if x=='high' else 2.0 if x=='low' else 3.0 )

    numerical_features = x.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = x.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        MLPClassifier(activation='logistic', hidden_layer_sizes=(256,), max_iter=1000)
    )
    best_score = 0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, x, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if best_score < score.mean():
            best_score = score.mean()
            best_pipe = pipe

    joblib.dumb(best_pipe, 'loan.pipe')
if __name__ == '__main__':
    main()
