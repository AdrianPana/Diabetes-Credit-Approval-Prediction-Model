import pandas as pd
import matplotlib.pyplot as plt
import analysis 
import preprocess
import random_forest
import mlp
from utils import get_dataset_info, DIABETES, CREDIT

def solve_dataset(dataset, show_analysis = True, algos=[True, True, True, True], metrics = True):
    """
    Solve chosen dataset
    
    Args:
        dataset (str):
            Dataset chosen can only be DIABETES/CREDIT
        analysis (bool):
            Display plots about the dataset
        algos (list[bool]):
            Choose which algorithms to be applied
            0: Scikit Random Forest
            1: Lab Random Forest
            2: Scikit MLP
            3: Lab MLP
    """
    numeric, categorical, target, labels, class_weights, path, new_numeric, new_categorical = get_dataset_info(dataset=dataset)

    df = pd.read_csv(f"./tema2_{path}_full.csv")
    df_train = pd.read_csv(f"./tema2_{path}_train.csv")
    df_test = pd.read_csv(f"./tema2_{path}_test.csv")
    
    if show_analysis:
        analysis.analyze_attributes(df, numeric, categorical)
        analysis.analyze_class_equilibrium(df_train, df_test, target)
        analysis.analyze_correlation(df, numeric, categorical)

    df_train = preprocess.remove_outliers(df_train, numeric)
    df_test = preprocess.remove_outliers(df_test, numeric)

    df_train = preprocess.fill_missing_values(df_train, numeric, categorical)
    df_test = preprocess.fill_missing_values(df_test, numeric, categorical)
    
    if show_analysis:
        analysis.analyze_correlation(df_train, new_numeric, new_categorical)

    df_train = preprocess.scale_features(df_train, new_numeric)
    df_test = preprocess.scale_features(df_test, new_numeric)

    if show_analysis:
        analysis.analyze_attributes(df_train, new_numeric, new_categorical)

    results = []

    if algos[0]:
        accuracy, precision, recall, f1 = random_forest.scikit_random_forest(df_train, new_numeric, 
                                                                             new_categorical, target, labels, df_test, class_weights)
        results.append({"algo": "Scikit Random Forest",
                        "accuracy": accuracy,
                        **{f'Precision_{cls}': p for cls, p in zip(labels, precision)},
                        **{f'Recall_{cls}': r for cls, r in zip(labels, recall)},
                        **{f'F1_{cls}': f for cls, f in zip(labels, f1)}})
    if algos[1]:
        accuracy, precision, recall, f1 = random_forest.lab_random_forest(df_train, new_numeric,
                                                                           new_categorical, target, df_test, labels)
        results.append({"algo": "Lab Random Forest",
                        "accuracy": accuracy,
                        **{f'Precision_{cls}': p for cls, p in zip(labels, precision)},
                        **{f'Recall_{cls}': r for cls, r in zip(labels, recall)},
                        **{f'F1_{cls}': f for cls, f in zip(labels, f1)}})
    if algos[2]:
        accuracy, precision, recall, f1 = mlp.scikit_mlp(df_train, new_numeric,
                                                        new_categorical, target, df_test, labels)
        results.append({"algo": "Scikit MLP",
                        "accuracy": accuracy,
                        **{f'Precision_{cls}': p for cls, p in zip(labels, precision)},
                        **{f'Recall_{cls}': r for cls, r in zip(labels, recall)},
                        **{f'F1_{cls}': f for cls, f in zip(labels, f1)}})
    if algos[3]:
        accuracy, precision, recall, f1 = mlp.lab_mlp(df_train, new_numeric,
                                                        new_categorical, target, df_test, labels)
        results.append({"algo": "Lab MLP",
                        "accuracy": accuracy,
                        **{f'Precision_{cls}': p for cls, p in zip(labels, precision)},
                        **{f'Recall_{cls}': r for cls, r in zip(labels, recall)},
                        **{f'F1_{cls}': f for cls, f in zip(labels, f1)}})

    if metrics:
        df_results = pd.DataFrame(results)
        print(df_results)

    

if __name__ == "__main__":
    solve_dataset(DIABETES, show_analysis=False, algos=[True, True, True, True])
    solve_dataset(CREDIT, show_analysis=False, algos=[True, True, True, True])

