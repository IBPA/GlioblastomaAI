from itertools import product
import pickle

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score,
    confusion_matrix,
)
from imblearn.metrics import specificity_score
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

from ...utils import load_dataset, initialize_best_model


def _load_metrics(path_file):
    with open(path_file, 'rb') as f:
        best_model_predictions = pickle.load(f)
    y_true = best_model_predictions['y_tests']
    y_pred = best_model_predictions['y_preds']
    y_score = best_model_predictions['y_scores']

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

    return result


if __name__ == '__main__':
    model = initialize_best_model()
    inputs_train, labels_train = load_dataset(train=True)
    inputs_test, labels_test = load_dataset(train=False)

    metrics_vnl = _load_metrics(
        "outputs/classification/best_model_predictions.pkl"
    )
    metrics_vnl_mean = metrics_vnl.groupby(['metric', 'avg_method']).mean()
    metrics_vnl_std = metrics_vnl.groupby(['metric', 'avg_method']).std()
    metrics_vnl_summary = pd.concat(
        [metrics_vnl_mean, metrics_vnl_std],
        axis=1
    ).drop(columns=['run_id'])
    metrics_vnl_summary.columns = ['mean', 'std']
    print("Best model vanilla:")
    print(metrics_vnl_summary)
    print()

    metrics_sfs = _load_metrics(
        "outputs/classification/best_model_sfs_predictions.pkl"
    )
    metrics_sfs_mean = metrics_sfs.groupby(['metric', 'avg_method']).mean()
    metrics_sfs_std = metrics_sfs.groupby(['metric', 'avg_method']).std()
    metrics_sfs_summary = pd.concat(
        [metrics_sfs_mean, metrics_sfs_std],
        axis=1
    ).drop(columns=['run_id'])
    metrics_sfs_summary.columns = ['mean', 'std']
    print("Best model SFS:")
    print(metrics_sfs_summary)
    print()

    for metric, avg_method in product(
                metrics_vnl['metric'].unique(),
                metrics_vnl['avg_method'].unique(),
            ):
        print(metric, avg_method)
        scores_vnl = metrics_vnl.query(
            f"metric == '{metric}' & avg_method == '{avg_method}'"
        )['score'].values
        scores_sfs = metrics_sfs.query(
            f"metric == '{metric}' & avg_method == '{avg_method}'"
        )['score'].values

        print(ttest_ind(scores_vnl, scores_sfs))
