import os

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from ..utils import load_dataset, load_cv_splits, get_idx_splits


def tune_logistic_regression(inputs, labels, splits_idx):
    gscv = GridSearchCV(
        estimator=Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('cls', LogisticRegression()),
            ]
        ),
        param_grid={
            'cls__penalty': ['l2', None],
            'cls__C': [0.01, 0.1, 1],
            'cls__solver': ['lbfgs'],
            'cls__max_iter': [100, 500, 1000],
            'cls__random_state': [1, 2, 3, 4, 5],
        },
        cv=splits_idx,
        n_jobs=-1,
        return_train_score=True,
    ).fit(inputs, labels)
    gscv_results = pd.DataFrame(gscv.cv_results_).sort_values(
        'mean_test_score', ascending=False
    )
    gscv_results.to_csv(
        "outputs/classification/gscv_results_lr.csv", index=False
    )


def tune_svm(inputs, labels, splits_idx):
    gscv = GridSearchCV(
        estimator=Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('cls', SVC()),
            ]
        ),
        param_grid={
            'cls__C': [0.01, 0.1, 1],
            'cls__kernel': ['linear', 'poly', 'rbf'],
            'cls__random_state': [6, 7, 8, 9, 10],
        },
        cv=splits_idx,
        n_jobs=-1,
        return_train_score=True,
    ).fit(inputs, labels)
    gscv_results = pd.DataFrame(gscv.cv_results_).sort_values(
        'mean_test_score', ascending=False
    )
    gscv_results.to_csv(
        "outputs/classification/gscv_results_svm.csv", index=False
    )


def tune_random_forest(inputs, labels, splits_idx):
    gscv = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid={
            'n_estimators': [100, 500],
            'max_depth': [2, 3, 4, 5, 6, 7],
            'random_state': [11, 12, 13, 14, 15],
        },
        cv=splits_idx,
        n_jobs=-1,
        return_train_score=True,
    ).fit(inputs, labels)
    gscv_results = pd.DataFrame(gscv.cv_results_).sort_values(
        'mean_test_score', ascending=False
    )
    gscv_results.to_csv(
        "outputs/classification/gscv_results_rf.csv", index=False
    )


def tune_adaboost(inputs, labels, splits_idx):
    gscv = GridSearchCV(
        estimator=AdaBoostClassifier(),
        param_grid={
            'n_estimators': [25, 50, 100, 500],
            'learning_rate': [0.01, 0.1, 1],
            'random_state': [16, 17, 18, 19, 20],
        },
        cv=splits_idx,
        n_jobs=-1,
        return_train_score=True,
    ).fit(inputs, labels)
    gscv_results = pd.DataFrame(gscv.cv_results_).sort_values(
        'mean_test_score', ascending=False
    )
    gscv_results.to_csv(
        "outputs/classification/gscv_results_ab.csv", index=False
    )


def tune_mlp(inputs, labels, splits_idx):
    gscv = GridSearchCV(
        estimator=Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('cls', MLPClassifier()),
            ]
        ),
        param_grid={
            'cls__hidden_layer_sizes': [(200,), (100,), (100, 100)],
            'cls__learning_rate': ['constant', 'adaptive'],
            'cls__learning_rate_init': [0.001, 0.01, 0.1],
            'cls__early_stopping': [True],
            'cls__random_state': [21, 22, 23, 24, 25],
        },
        cv=splits_idx,
        n_jobs=-1,
        return_train_score=True,
    ).fit(inputs, labels)
    gscv_results = pd.DataFrame(gscv.cv_results_).sort_values(
        'mean_test_score', ascending=False
    )
    gscv_results.to_csv(
        "outputs/classification/gscv_results_mlp.csv", index=False
    )


def summarize_model_selection_results():
    results_rows = []
    for cls in ['ab', 'lr', 'mlp', 'rf', 'svm']:
        result = pd.read_csv(f"outputs/classification/gscv_results_{cls}.csv")
        params = [
            c for c in result.columns
            if 'param_' in c and 'random_state' not in c
        ]

        def get_results_row(group, results_rows):
            results_rows += [{
                'classifier': cls,
                'mean_train_score': group['mean_train_score'].mean(),
                'std_train_score': group['mean_train_score'].std(),
                'mean_test_score': group['mean_test_score'].mean(),
                'std_test_score': group['mean_test_score'].std(),
                'params': group['params'].iloc[0],
            }]
        result.groupby(params).apply(
            lambda group: get_results_row(group, results_rows)
        )
    results = pd.DataFrame(results_rows).sort_values(
        'mean_test_score', ascending=False
    )
    results.to_csv(
        "outputs/classification/gscv_results_summary.csv", index=False
    )


if __name__ == '__main__':
    os.makedirs("outputs/classification", exist_ok=True)

    inputs_train, labels_train = load_dataset(train=True)
    inputs_test, labels_test = load_dataset(train=False)
    splits = load_cv_splits()
    splits_idx = get_idx_splits(splits, inputs_train)

    tune_logistic_regression(inputs_train, labels_train, splits_idx)
    tune_svm(inputs_train, labels_train, splits_idx)
    tune_random_forest(inputs_train, labels_train, splits_idx)
    tune_adaboost(inputs_train, labels_train, splits_idx)
    tune_mlp(inputs_train, labels_train, splits_idx)

    summarize_model_selection_results()
