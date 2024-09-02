import pandas as pd
from scipy.stats import ttest_ind


if __name__ == '__main__':
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

    # Perform t-test.
    scores_rf = data_plot.query("classifier == 'rf'")['test_score'].values
    scores_ab = data_plot.query("classifier == 'ab'")['test_score'].values
    scores_svm = data_plot.query("classifier == 'svm'")['test_score'].values
    scores_lr = data_plot.query("classifier == 'lr'")['test_score'].values
    scores_mlp = data_plot.query("classifier == 'mlp'")['test_score'].values

    print(f"RF : {scores_rf.mean()} +/- {scores_rf.std()}")
    print(f"AB : {scores_ab.mean()} +/- {scores_ab.std()}")
    print(f"SVM: {scores_svm.mean()} +/- {scores_svm.std()}")
    print(f"LR : {scores_lr.mean()} +/- {scores_lr.std()}")
    print(f"MLP: {scores_mlp.mean()} +/- {scores_mlp.std()}")

    print(f"RF vs. AB : p={ttest_ind(scores_rf, scores_ab)[1]}")
    print(f"RF vs. SVM: p={ttest_ind(scores_rf, scores_svm)[1]}")
    print(f"AB vs. SVM: p={ttest_ind(scores_ab, scores_svm)[1]}")
