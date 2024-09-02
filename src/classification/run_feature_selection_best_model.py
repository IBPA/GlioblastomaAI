import pickle

import pandas as pd

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from ..utils import (
    load_dataset, load_cv_splits, get_idx_splits, initialize_best_model
)


def run_dump_sfs(model, inputs, labels, splits_idx, path_save_dir):
    sfs = SFS(
        estimator=model,
        k_features='parsimonious',
        forward=False,
        floating=False,
        scoring='accuracy',
        cv=splits_idx,
        n_jobs=-1,
        verbose=2,
    ).fit(inputs, labels)

    with open(f"{path_save_dir}/sfs.pkl", 'wb') as f:
        pickle.dump(sfs, f)

    pd.DataFrame.from_dict(sfs.get_metric_dict()).T.to_csv(
        f"{path_save_dir}/sfs_results.csv", index=False
    )


if __name__ == '__main__':
    inputs_train, labels_train = load_dataset(train=True)
    inputs_test, labels_test = load_dataset(train=False)
    splits = load_cv_splits()
    splits_idx = get_idx_splits(splits, inputs_train)

    model = initialize_best_model()
    print(model)

    # Feature selection.
    run_dump_sfs(
        model,
        inputs_train,
        labels_train,
        splits_idx,
        path_save_dir="outputs/classification",
    )
