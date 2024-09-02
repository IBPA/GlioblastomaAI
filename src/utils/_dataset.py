import pandas as pd


def easy_preprocess_inputs(inputs, labels):
    # Drop the sample with too many missing values.
    inputs = inputs.drop(index=['Pt_34_3_24'])
    labels = labels.drop(index=['Pt_34_3_24'])

    # Apply zero-imputation.
    inputs = inputs.fillna(0)

    return inputs, labels


def load_data(prep=True):
    """Load the dataset from the excel file.
    Args:
        None

    Returns:
        (pd.DataFrame, pd.Series): The inputs and labels.

    """
    data = pd.read_excel(
        "data/Biogenic Amines Final Dataset for Processing.xlsx",
        sheet_name="Sheet2",
    )
    feature_names = data.columns.tolist()[2:]
    data[['sort_idx_1', 'sort_idx_2']] = data['sample_id'].apply(
        lambda x: pd.Series(x.split('_')[1:3]).astype(int)
    )
    data = data.sort_values(['sort_idx_1', 'sort_idx_2'])
    data['label'] = data['treatment_status'].str.strip().map({
        'Pre-Surgery': 0,
        'Post-Surgery': 1,
        'Pre-Radiation': 2,
        'Post-Radiation_1': 3,
        'Post-Radiation_2': 3,
        'Post-Radiation_3': 3,
        'Post-Radiation_4': 3,
    })
    data = data.set_index(['sample_id'])

    inputs = data[feature_names].astype(float)
    labels = data['label']

    if prep:
        inputs, labels = easy_preprocess_inputs(inputs, labels)

    return inputs, labels


def load_dataset(train=True):
    if train:
        inputs = pd.read_csv(
            "outputs/data_processing/inputs_train.csv",
            index_col='sample_id',
        )
        labels = pd.read_csv(
            "outputs/data_processing/labels_train.csv",
            index_col='sample_id',
        )['label']
    else:
        inputs = pd.read_csv(
            "outputs/data_processing/inputs_test.csv",
            index_col='sample_id',
        )
        labels = pd.read_csv(
            "outputs/data_processing/labels_test.csv",
            index_col='sample_id',
        )['label']

    return inputs, labels


def load_cv_splits():
    PATH_DATA_DIR = "outputs/data_processing/splits"

    splits = []
    for i in range(5):
        input_train = pd.read_csv(
            f"{PATH_DATA_DIR}/{i}/inputs_train.csv",
            index_col='sample_id',
        )
        input_val = pd.read_csv(
            f"{PATH_DATA_DIR}/{i}/inputs_val.csv",
            index_col='sample_id',
        )
        label_train = pd.read_csv(
            f"{PATH_DATA_DIR}/{i}/labels_train.csv",
            index_col='sample_id',
        )['label']
        label_val = pd.read_csv(
            f"{PATH_DATA_DIR}/{i}/labels_val.csv",
            index_col='sample_id',
        )['label']

        splits += [[input_train, input_val, label_train, label_val]]

    return splits


def get_idx_splits(splits, inputs):
    inputs = inputs.reset_index()
    idxs_splits = []
    for split in splits:
        idxs_train_str = split[0].index
        idxs_val_str = split[1].index

        assert len(set(idxs_train_str).intersection(set(idxs_val_str))) == 0

        idxs_train = inputs[
            inputs['sample_id'].isin(idxs_train_str)].index.tolist()
        idxs_val = inputs[
            inputs['sample_id'].isin(idxs_val_str)].index.tolist()
        idxs_splits += [(idxs_train, idxs_val)]

    return idxs_splits
