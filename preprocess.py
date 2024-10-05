from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def fill_missing_values(df, numeric, categorical):

    numeric_strategy = 'mean'
    categorical_strategy = 'most_frequent'

    imputer = SimpleImputer(strategy=numeric_strategy)
    df[numeric] = imputer.fit_transform(df[numeric])

    imputer = SimpleImputer(strategy=categorical_strategy)
    df[categorical] = imputer.fit_transform(df[categorical])

    return df

def remove_outliers(df, numeric):
    for col in numeric:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        threshold = 1
        bound1 = Q1 - threshold * IQR
        bound2 = Q3 + threshold * IQR
        df[col] = np.where((df[col] < bound1) | (df[col] > bound2), None, df[col])
    
    return df

def scale_features(df, numeric):

    scaler = StandardScaler()

    df[numeric] = scaler.fit_transform(df[numeric])

    return df
