import os
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from ..utils import load_dataset


def get_pca(inputs, labels=None):
    """Run PCA with one component.

    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(
            n_components=1,
            random_state=42,
        )),
    ])
    model.fit(inputs)
    pca_result = pd.DataFrame(
        model['pca'].components_[0],
        index=inputs.columns,
        columns=['val'],
    )
    pca_result['rank'] = pca_result.abs().rank(ascending=False)

    return pca_result


def get_lda(inputs, labels):
    """Run LDA with one component.

    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(
            n_components=1,
            solver='eigen',
            shrinkage='auto',
        )),
    ])
    model.fit(inputs, labels)
    lda_result = pd.DataFrame(
        model['lda'].coef_.T,
        index=inputs.columns,
        columns=[f"val_cls{c}" for c in model['lda'].classes_],
    )
    lda_result['val'] = lda_result.abs().mean(axis=1)
    lda_result['rank'] = lda_result['val'].rank(ascending=False)

    return lda_result


def get_sfs(inputs=None, labels=None):
    """Load SFS result.

    """
    with open("outputs/classification/sfs.pkl", 'rb') as f:
        sfs_result = pickle.load(f)

    sfs_result \
        = pd.DataFrame.from_dict(sfs_result.subsets_).T['feature_names']
    for i in range(340 - 1):
        sfs_result.iloc[i] \
            = list(
                set(sfs_result.iloc[i]) - set(sfs_result.iloc[i + 1])
            )[0]
    sfs_result.iloc[-1] = list(sfs_result.iloc[-1])[0]
    sfs_result = sfs_result.reset_index().set_index('feature_names')
    sfs_result.columns = ['rank']
    sfs_result = sfs_result.loc[inputs.columns]

    return sfs_result


def get_rf(inputs, labels):
    rf_result_columns = []
    for rs in range(10):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=rs,
        ).fit(inputs, labels)

        rf_result_column = pd.Series(
            model.feature_importances_,
            index=inputs.columns,
        )
        rf_result_columns += [rf_result_column]
    rf_result = pd.concat(rf_result_columns, axis=1).mean(axis=1).to_frame()
    rf_result.columns = ['val']
    rf_result['rank'] = rf_result.rank(ascending=False)

    return rf_result


def get_rf_sfs(inputs, labels):
    with open("outputs/classification/sfs.pkl", 'rb') as f:
        sfs_result = pickle.load(f)

    inputs_sfs = inputs[list(sfs_result.k_feature_names_)]

    rf_sfs_result_columns = []
    for rs in range(10):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=rs,
        ).fit(inputs_sfs, labels)

        rf_sfs_result_column = pd.Series(
            model.feature_importances_,
            index=inputs_sfs.columns,
        )
        rf_sfs_result_columns += [rf_sfs_result_column]
    rf_sfs_result \
        = pd.concat(rf_sfs_result_columns, axis=1).mean(axis=1).to_frame()
    rf_sfs_result.columns = ['val']
    rf_sfs_result['rank'] = rf_sfs_result.rank(ascending=False)

    result = pd.DataFrame(
        index=inputs.columns,
        columns=['val', 'rank'],
    )
    result.loc[rf_sfs_result.index, ['val', 'rank']] = rf_sfs_result

    return result


if __name__ == '__main__':
    os.makedirs("outputs/feature_importance", exist_ok=True)

    inputs_train, labels_train = load_dataset(train=True)
    inputs_test, labels_test = load_dataset(train=False)
    inputs = pd.concat([inputs_train, inputs_test])
    labels = pd.concat([labels_train, labels_test])

    fi_pca = get_pca(inputs, labels)
    fi_lda = get_lda(inputs, labels)
    fi_sfs = get_sfs(inputs, labels)
    fi_rf = get_rf(inputs, labels)
    fi_rf_sfs = get_rf_sfs(inputs, labels)

    fi_pca.to_csv("outputs/feature_importance/pca.csv")
    fi_lda.to_csv("outputs/feature_importance/lda.csv")
    fi_sfs.to_csv("outputs/feature_importance/sfs.csv")
    fi_rf.to_csv("outputs/feature_importance/rf.csv")
    fi_rf_sfs.to_csv("outputs/feature_importance/rf_sfs.csv")
