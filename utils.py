import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

DIABETES = "Diabetes"
CREDIT = "Credit"
CREDIT_TARGET = "loan_approval_status"

def get_dataset_info(dataset):
    """
    Return dataset info used in future analysis and processing
    
    Args:
        dataset (str):
            Dataset chosen can only be "Diabetes"/"Credit"
    """

    if (dataset == 'Diabetes'):
        numeric = ["psychological-rating", "BodyMassIndex", "Age", "CognitionScore", "Body_Stats", "Metabolical_Rate"]
        categorical = ["HealthcareInterest", "PreCVA", "RoutineChecks", "CompletedEduLvl", "alcoholAbuse", "cholesterol_ver",
                        "vegetables", "HighBP", "Unprocessed_fructose", "Jogging", "IncreasedChol", "gender", "myocardial_infarction",
                        "SalaryBraket", "Cardio", "ImprovedAveragePulmonaryCapacity", "Smoker"]
        target = "Diabetes"
        labels=[0.0, 1.0, 2.0]
        class_weights = {0.0: 0.87, 1.0: 0.02, 2.0: 0.11}
        path = "Diabet/Diabet"
        new_numeric = ["psychological-rating", "BodyMassIndex", "Age", "CognitionScore"]
        new_categorical = ["HealthcareInterest", "PreCVA", "gender"]

        return numeric, categorical, target, labels, class_weights, path, new_numeric, new_categorical
    
    elif (dataset == 'Credit'):
        numeric = ["applicant_age", "applicant_income", "job_tenure_years", "loan_amount", "loan_rate",
                    "loan_income_ratio", "credit_history_length_years", "credit_history_length_months"]
        categorical = ["residential_status", "loan_purpose", "loan_rating", "credit_history_default_status", "stability_rating"]
        target = "loan_approval_status"
        labels = [1, 0] # as opposed to ["Approved", "Declined"]
        class_weights = {"Approved": 0.78, "Declined": 0.22}
        path = "Credit_Risk/credit_risk"
        new_numeric = ["applicant_age", "applicant_income", "job_tenure_years", "loan_amount", "loan_rate",
                    "loan_income_ratio"]
        new_categorical = ["residential_status"]

        return numeric, categorical, target, labels, class_weights, path, new_numeric, new_categorical
    else:
        raise ValueError("Unknown dataset, try DIABETES/CREDIT")

def prepare_data(df_train, numeric, categorical, target, df_test):
    attributes = numeric + categorical
    X_train = df_train[attributes]
    y_train = df_train[target]

    X_test = df_test[attributes]
    y_test = df_test[target]

    X_train = pd.get_dummies(X_train, columns=categorical, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical, drop_first=True)

    X_train = X_train.reindex(columns=X_test.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    if target == CREDIT_TARGET:
        y_train = y_train.map({'Approved': 1, 'Declined': 0})
        y_test = y_test.map({'Approved': 1, 'Declined': 0})

    return X_train, y_train, X_test, y_test

def gini_index(groups, y, classes):
    n_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            count = (y == class_val).sum()
            p =  count / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def test_split(dataset, feature, value):
    return dataset[dataset[feature] <= value], dataset[dataset[feature] > value]

def get_best_split_value(dataset, feature, att_values, y):
    class_values = y.unique()
    b_value, b_score= 999, 999
    for value in att_values:
        groups = test_split(dataset, feature, value)
        gini = gini_index(groups, y, class_values)
        if gini < b_score:
            b_value, b_score = value, gini
    return b_value

def plot_confusion_matrix(y_pred: pd.Series, y_true: pd.Series):

    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)

    all_classes = np.unique(list(y_true) + list(y_pred))
    confusion_matrix = confusion_matrix.reindex(index=all_classes, columns=all_classes, fill_value=0)
    
    plt.figure(figsize=(10,7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

def get_metrics(y_test, predictions, labels):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    r_accuracy = accuracy_score(y_test, predictions)
    r_precision, r_recall, r_f1, _ = precision_recall_fscore_support(y_test, predictions, labels=labels, zero_division=0)

    return r_accuracy, r_precision, r_recall, r_f1