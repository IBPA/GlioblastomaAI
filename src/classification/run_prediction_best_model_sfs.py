import pickle

from tqdm import tqdm

from ..utils import load_dataset, initialize_best_model


def _get_best_model_predictions(
    model,
    inputs_train,
    labels_train,
    inputs_test,
    labels_test,
    n_runs,
):
    # Concatenate the results from multiple runs.
    y_tests = []
    y_preds = []
    y_scores = []
    for rs in tqdm(range(n_runs)):
        model.set_params(random_state=rs)
        model.fit(inputs_train, labels_train)
        y_pred = model.predict(inputs_test)
        y_score = model.predict_proba(inputs_test)
        y_tests += [labels_test]
        y_preds += [y_pred]
        y_scores += [y_score]

    return y_tests, y_preds, y_scores


if __name__ == '__main__':
    model = initialize_best_model()
    inputs_train, labels_train = load_dataset(train=True)
    inputs_test, labels_test = load_dataset(train=False)
    N_RUNS = 20

    # Test the best model with SFS.
    with open("outputs/classification/sfs.pkl", 'rb') as f:
        sfs = pickle.load(f)

    inputs_train_sfs = inputs_train[list(sfs.k_feature_names_)]
    inputs_test_sfs = inputs_test[list(sfs.k_feature_names_)]

    y_tests, y_preds, y_scores = _get_best_model_predictions(
        model, inputs_train_sfs, labels_train.values, inputs_test_sfs,
        labels_test.values, N_RUNS
    )

    with open(
            "outputs/classification/best_model_sfs_predictions.pkl",
            'wb') as f:
        pickle.dump(
            {
                'y_tests': y_tests,
                'y_preds': y_preds,
                'y_scores': y_scores,
            },
            f,
        )
