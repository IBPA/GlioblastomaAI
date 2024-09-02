import os

import numpy as np
import pandas as pd

from ..utils import load_data


def split_train_test(
        inputs,
        labels,
        test_size=0.2,
        tol=0.075,
        random_state=42,
        ):
    """Split train and test sets such that they are stratified while patients
    are not shared among the two sets.

    """
    # The ideal label distribution in the test set.
    test_split_dist_gt = (labels.value_counts() * test_size).sort_index()
    print(test_split_dist_gt)

    rng = np.random.default_rng(random_state)
    patients = pd.Series(
        [int(x.split('_')[1]) for x in inputs.index.tolist()],
        index=inputs.index,
    )

    # We will try 1000 times to find if such a split exists.
    for i in range(1000):
        print("Trial:", i + 1)
        restart = False
        patient_ids_test = []
        patient_ids_permuted = rng.permutation(np.unique(patients))

        for patient_id in patient_ids_permuted:
            patient_ids_test += [patient_id]
            test_split = labels[patients.isin(patient_ids_test)]
            test_split_dist = test_split.value_counts().sort_index()

            # If the test set is stratified within the error tolerance,
            #   returns.
            if ((test_split_dist - test_split_dist_gt).abs()
                    < test_split_dist_gt * tol).all():
                print("Found!")
                inputs_train, inputs_test, labels_train, labels_test \
                    = inputs[~patients.isin(patient_ids_test)], \
                    inputs[patients.isin(patient_ids_test)], \
                    labels[~patients.isin(patient_ids_test)], \
                    labels[patients.isin(patient_ids_test)]

                return inputs_train, inputs_test, labels_train, labels_test

            # If the test is not within the error tolerance but smaller than
            #   the ideal distribution, add more patients.
            elif ((test_split_dist - test_split_dist_gt)
                    <= test_split_dist_gt * tol).any():
                # Smaller than the tolerance, add more patients.
                print("Add more patients.")
                continue

            # If the test is not within the error tolerance but larger than
            #   the ideal distribution, restart.
            else:
                # Exceeds the tolerance, restart.
                print(test_split_dist)
                print("Exceeds the tolerance, restart.\n")
                restart = True
                break

        if restart:
            continue


if __name__ == '__main__':
    inputs, labels = load_data()
    inputs_train, inputs_test, labels_train, labels_test \
        = split_train_test(inputs, labels)

    os.makedirs("outputs/data_processing", exist_ok=True)
    inputs_train.to_csv(
        "outputs/data_processing/inputs_train.csv",
    )
    inputs_test.to_csv(
        "outputs/data_processing/inputs_test.csv",
    )
    labels_train.to_csv(
        "outputs/data_processing/labels_train.csv",
    )
    labels_test.to_csv(
        "outputs/data_processing/labels_test.csv",
    )
