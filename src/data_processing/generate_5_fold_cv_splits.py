import os
from decimal import localcontext, Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

from ..utils import load_dataset


def split_5_fold_splits(
        inputs,
        labels,
        random_state=42,
        ):
    """
    """
    rng = np.random.default_rng(random_state)

    # The ideal label distribution in the test set.
    split_dist_gt = (labels.value_counts() / 5).sort_index()

    patients = pd.Series(
        [int(x.split('_')[1]) for x in inputs.index.tolist()],
        index=inputs.index,
    )
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        n_patient_split = int(
            Decimal(patients.nunique() / 5).to_integral_value()
        )
    print(split_dist_gt)

    # We will try 100 times to find the split that has minimum difference.
    diff_best = np.inf
    patient_ids_splits_best = None
    for i in range(1000):
        print("Trial:", i + 1)
        patient_ids_permuted = rng.permutation(np.unique(patients))
        patient_ids_splits = []

        for j in range(5):
            patient_ids_splits += [
                patient_ids_permuted[
                    j * n_patient_split:(j + 1) * n_patient_split
                ]
            ]

        diff = 0
        for patient_ids_split in patient_ids_splits:
            labels_split = labels[patients.isin(patient_ids_split)]
            split_dist = labels_split.value_counts().sort_index()
            diff += (split_dist - split_dist_gt).abs().sum()
        diff = diff / 5
        if diff < diff_best:
            diff_best = diff
            patient_ids_splits_best = patient_ids_splits

    inputs_splits = []
    labels_splits = []
    for patient_ids in patient_ids_splits_best:
        inputs_splits += [inputs[patients.isin(patient_ids)]]
        labels_splits += [labels[patients.isin(patient_ids)]]

    splits = []
    for i in range(5):
        split_id_val = i
        split_id_train = [x for x in list(range(5)) if x != i]

        inputs_val = inputs_splits[split_id_val]
        labels_val = labels_splits[split_id_val]
        inputs_train = pd.concat(
            [inputs_splits[x] for x in split_id_train],
            axis=0,
        )
        labels_train = pd.concat(
            [labels_splits[x] for x in split_id_train],
            axis=0,
        )
        splits += [[inputs_train, inputs_val, labels_train, labels_val]]

    return splits


if __name__ == '__main__':
    inputs, labels = load_dataset(train=True)
    splits = split_5_fold_splits(inputs, labels)
    for i, split in enumerate(splits):
        os.makedirs(f"outputs/data_processing/splits/{i}", exist_ok=True)
        split[0].to_csv(
            f"outputs/data_processing/splits/{i}/inputs_train.csv",
        )
        split[1].to_csv(
            f"outputs/data_processing/splits/{i}/inputs_val.csv",
        )
        split[2].to_csv(
            f"outputs/data_processing/splits/{i}/labels_train.csv",
        )
        split[3].to_csv(
            f"outputs/data_processing/splits/{i}/labels_val.csv",
        )
