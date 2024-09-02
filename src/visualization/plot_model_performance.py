import os
import pickle

from scipy.stats import ttest_ind
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score,
    confusion_matrix,
)
from imblearn.metrics import specificity_score
from mlxtend.plotting import (
    plot_confusion_matrix,
    plot_sequential_feature_selection as plot_sfs,
)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _plot_clfs(path_save=None):
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
                'mean_test_score': group['mean_test_score'].mean(),
                'test_scores': group[
                    [f"split{i}_test_score" for i in range(5)]
                ].values,
                'params': group['params'].iloc[0],
            }]
        result.groupby(params).apply(
            lambda group: get_results_row(group, results_rows)
        )
    results = pd.DataFrame(results_rows).sort_values(
        'mean_test_score', ascending=False
    )
    results = results.groupby('classifier').apply(
        lambda group: group.iloc[0]
    ).sort_values('mean_test_score', ascending=False)

    def get_data_plot_row(row, data_plot_rows):
        for i, test_scores in enumerate(row['test_scores']):
            for j, test_score in enumerate(test_scores):
                data_plot_rows += [{
                    'classifier': row['classifier'],
                    'run': i,
                    'split': j,
                    'test_score': test_score,
                }]

    data_plot_rows = []
    results.apply(lambda row: get_data_plot_row(row, data_plot_rows), axis=1)
    data_plot = pd.DataFrame(data_plot_rows)

    sns.set_style('whitegrid')
    plt.figure(figsize=(6.4, 6.4))
    g = sns.boxplot(
        x='classifier',
        y='test_score',
        data=data_plot,
        palette='Set3',
    )
    g.set_ylim(0.3, 1)
    plt.savefig(path_save)
    plt.close()


def _plot_metrics(y_true, y_pred, path_save=None):
    result_rows = []
    for ir in range(len(y_true)):
        for avg_method in ['micro', 'macro']:
            for metric in [
                    'accuracy',
                    'f1',
                    'precision',
                    'recall',
                    'specificity',
                    ]:
                if metric == 'accuracy':
                    score = eval(
                        "accuracy_score(y_true[ir], y_pred[ir])"
                    ) if avg_method == 'micro' else eval(
                        "balanced_accuracy_score(y_true[ir], y_pred[ir])"
                    )
                else:
                    score = eval(
                        f"{metric}_score(y_true[ir], y_pred[ir], "
                        f"average='{avg_method}')"
                    )
                result_rows += [{
                    'run_id': ir,
                    'metric': metric,
                    'avg_method': avg_method,
                    'score': score,
                }]

    result = pd.DataFrame(result_rows)

    plt.figure(figsize=(6.4, 6.4))
    sns.set_style('whitegrid')
    sns.boxplot(
        x='metric',
        y='score',
        hue='avg_method',
        data=result,
    )
    plt.savefig(path_save)
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, path_save=None):
    cms = []
    for i in range(len(y_true)):
        cm = confusion_matrix(y_true[i], y_pred[i])
        cms += [cm]
    cm_mean = np.mean(cms, axis=0)
    cm_std = np.std(cms, axis=0)
    print(cm_mean)
    print(cm_std)

    plt.figure(figsize=(6.4, 6.4))
    plot_confusion_matrix(
        cm_mean,
        colorbar=True,
        class_names=[
            'Pre-surgery', 'Post-surgery', 'Pre-radiation', 'Post-radiation'
        ],
    )
    plt.savefig(path_save)
    plt.close()


def _plot_pr_curve(y_true, y_score, path_save=None):
    fig = plt.figure(figsize=(6.4, 6.4))

    # Plot the multi-class PR curve.
    for i, cls_name in zip(
            range(4),
            ['Pre-surgery', 'Post-surgery', 'Pre-radiation', 'Post-radiation']
            ):
        precisions, recalls, _ = precision_recall_curve(
            y_true[:, i], y_score[:, i]
        )
        average_precisions = average_precision_score(
            y_true[:, i], y_score[:, i]
        )
        plt.plot(
            recalls, precisions,
            label=f"{cls_name} (AUPRC={average_precisions:.2f})"
        )
    precisions, recalls, _ \
        = precision_recall_curve(
            y_true.ravel(), y_score.ravel()
        )
    average_precisions = average_precision_score(
        y_true.ravel(), y_score.ravel()
    )
    plt.plot(
        recalls, precisions,
        label=f"Micro Average (AUPRC={average_precisions:.2f})"
    )
    p_majority = y_true.sum(axis=0).max() / y_true.sum(axis=0).sum()

    plt.plot(
        [0, 1], [p_majority, p_majority],
        linestyle='--',
        color='black',
        label='Baseline (Majority Class)',
    )
    plt.title('One-vs-rest Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.savefig(path_save)
    plt.close()


def _plot_roc_curve(y_true, y_score, path_save=None):
    # Plot the multi-class ROC curve.
    fig = plt.figure(figsize=(6.4, 6.4))

    for i, cls_name in zip(
            range(4),
            ['Pre-surgery', 'Post-surgery', 'Pre-radiation', 'Post-radiation']
            ):
        fprs, tprs, _ = roc_curve(
            y_true[:, i], y_score[:, i]
        )
        auroc = roc_auc_score(
            y_true[:, i], y_score[:, i]
        )
        plt.plot(
            fprs, tprs,
            label=f"{cls_name} (AUROC={auroc:.2f})"
        )
    fprs, tprs, _ \
        = roc_curve(
            y_true.ravel(), y_score.ravel()
        )
    auroc = roc_auc_score(
        y_true.ravel(), y_score.ravel()
    )
    plt.plot(
        fprs, tprs,
        label=f"Micro Average (AUROC={auroc:.2f})"
    )
    plt.plot(
        [0, 1], [0, 1],
        linestyle='--',
        color='black',
        label='Baseline (Random)',
    )
    plt.title('One-vs-rest Receiver Operating Characteristic Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(path_save)
    plt.close()


def _plot_sfs(path_save=None):
    # Load the SFS result.
    with open("outputs/classification/sfs.pkl", 'rb') as f:
        sfs = pickle.load(f)

    g = plot_sfs(
        sfs.get_metric_dict(),
        kind='std_dev',
        marker=None,
    )[1]
    g.xaxis.set_tick_params(bottom=False)
    g.set_xticklabels(
        ['1']
        + [
            x
            if (i + 1) % 50 == 0 or (i + 2) == len(sfs.k_feature_names_)
            else ''
            for i, x in enumerate(g.get_xticklabels()[1:-1])
        ]
        + ['340'],
    )
    plt.savefig(path_save)
    plt.close()


if __name__ == '__main__':
    os.makedirs("outputs/visualization/model_performance", exist_ok=True)

    # Load predictions for the best model.
    with open("outputs/classification/best_model_predictions.pkl", 'rb') as f:
        best_model_predictions = pickle.load(f)
    y_tests = best_model_predictions['y_tests']
    y_preds = best_model_predictions['y_preds']
    y_scores = best_model_predictions['y_scores']

    y_tests_concat_ohe = OneHotEncoder(sparse_output=False).fit_transform(
        np.concatenate(y_tests).reshape(-1, 1)
    )
    y_preds_concat_ohe = OneHotEncoder(sparse_output=False).fit_transform(
        np.concatenate(y_preds).reshape(-1, 1)
    )
    y_scores_concat = np.concatenate(y_scores)

    _plot_clfs(
        path_save="outputs/visualization/model_performance/clf.svg",
    )
    _plot_pr_curve(
        y_tests_concat_ohe,
        y_scores_concat,
        path_save="outputs/visualization/model_performance/pr.svg",
    )
    _plot_roc_curve(
        y_tests_concat_ohe,
        y_scores_concat,
        path_save="outputs/visualization/model_performance/roc.svg",
    )
    _plot_metrics(
        y_tests,
        y_preds,
        path_save="outputs/visualization/model_performance/metrics.svg",
    )
    _plot_confusion_matrix(
        y_tests,
        y_preds,
        path_save="outputs/visualization/model_performance/cm.svg",
    )
    _plot_sfs(
        path_save="outputs/visualization/model_performance/sfs_results.svg",
    )
