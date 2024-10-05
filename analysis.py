import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

def plot_numeric_data(df, data):
    df_describe = df[data].describe()
    
    with pd.option_context("display.max_columns", None):
        print(df_describe)

    boxplot = df.boxplot(column=data)
    plt.title('Numeric Attributes')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.show()

def plot_categorical_data(df, data):
    for col in data:
        plt.figure()
        df[col].dropna().hist(bins=30)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

def analyze_attributes(df, numeric, categorical):
    plot_numeric_data(df, numeric)
    plot_categorical_data(df, categorical)

def analyze_class_equilibrium(df_train, df_test, target):

    train_value_counts = df_train[target].value_counts(normalize=True)
    test_value_counts = df_test[target].value_counts(normalize=True)

    print(f"Train target value percentages: {train_value_counts}")
    print(f"Test target value percentages: {test_value_counts}")

    plt.figure()
    plt.hist([df_train[target].dropna(), df_test[target].dropna()], bins=30)
    plt.title(f'Histogram of {target}')
    plt.xlabel(target)
    plt.ylabel('Frequency')
    plt.show()

def numeric_correlation(df, numeric):
    correlations = df[numeric].corr()
    print(correlations)

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)

    ticks = np.arange(0,len(numeric),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(numeric)
    ax.set_yticklabels(numeric)

    plt.show()

def categorical_correlation(df, categorical):
    correlations = pd.DataFrame(index=categorical, columns=categorical)

    for cat1 in categorical:
        for cat2 in categorical:
            if cat1 == cat2:
                correlations.loc[cat1, cat2] = 1.0
                continue
            else:
                CrosstabResult=pd.crosstab(index=df[cat1], columns=df[cat2])
                ChiSqResult = chi2_contingency(CrosstabResult)
                correlation_status = 0.0 if ChiSqResult[1] > 0.05 else 1.0
                correlations.loc[cat1, cat2] = correlation_status
                correlations.loc[cat2, cat1] = correlation_status

    fig = plt.figure(figsize=(20,20))

    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations.astype(float), vmin=0.0, vmax=1.0)
    fig.colorbar(cax)

    ticks = np.arange(0,len(categorical),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(categorical)
    ax.set_yticklabels(categorical)

    plt.show()

def analyze_correlation(df, numeric, categorical):
    numeric_correlation(df, numeric)
    categorical_correlation(df, categorical)