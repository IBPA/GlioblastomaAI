import os
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import shap
import numpy as np
import pandas as pd

from ..utils import load_dataset, initialize_best_model


def get_lda_component_weight(inputs, labels):
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


def get_rf_feature_importance(inputs, labels):
    """
    """
    model = initialize_best_model()

    fi_columns = []
    for rs in range(20):
        model.set_params(random_state=rs)
        model.fit(inputs, labels)

        fi_column = pd.Series(
            model.feature_importances_,
            index=inputs.columns,
        )
        fi_columns += [fi_column]
    fi = pd.concat(fi_columns, axis=1).mean(axis=1).to_frame()
    fi.columns = ['val']
    fi['rank'] = fi.rank(ascending=False)

    return fi


def get_rf_sfs_feature_importance(inputs, labels):
    """
    """
    model = initialize_best_model()
    with open("outputs/classification/sfs.pkl", 'rb') as f:
        sfs_result = pickle.load(f)
    inputs = inputs[list(sfs_result.k_feature_names_)]

    fi_columns = []
    for rs in range(20):
        model.set_params(random_state=rs)
        model.fit(inputs, labels)

        fi_column = pd.Series(
            model.feature_importances_,
            index=inputs.columns,
        )
        fi_columns += [fi_column]
    fi = pd.concat(fi_columns, axis=1).mean(axis=1).to_frame()
    fi.columns = ['val']
    fi['rank'] = fi.rank(ascending=False)

    return fi


def get_sfs_feature_rank(inputs=None, labels=None):
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


def get_rf_sfs_shap(inputs, labels):
    with open("outputs/classification/sfs.pkl", 'rb') as f:
        sfs_result = pickle.load(f)
    inputs = inputs[list(sfs_result.k_feature_names_)]

    model = initialize_best_model()
    model.fit(inputs, labels)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(inputs)

    result_rows = []
    for i in range(4):
        result_rows += [np.absolute(shap_values[i]).mean(axis=1).tolist()]
    result = pd.DataFrame(
        result_rows,
        index=[f"val_cls{i}" for i in range(4)],
        columns=inputs.columns,
    ).T
    result['val'] = result.mean(axis=1)
    result['rank'] = result['val'].rank(ascending=False)
    print(result.sort_values('rank'))
    return result


if __name__ == '__main__':
    os.makedirs("outputs/feature_analysis", exist_ok=True)

    inputs_train, labels_train = load_dataset(train=True)
    inputs_test, labels_test = load_dataset(train=False)
    inputs = pd.concat([inputs_train, inputs_test])
    labels = pd.concat([labels_train, labels_test])

    fi_lda = get_lda_component_weight(inputs, labels)
    fi_lda.to_csv("outputs/feature_analysis/lda_comp_weight.csv")

    fi_rf = get_rf_feature_importance(inputs, labels)
    fi_rf.to_csv("outputs/feature_analysis/rf_feature_importance.csv")

    fi_rf_sfs = get_rf_sfs_feature_importance(inputs, labels)
    fi_rf_sfs.to_csv("outputs/feature_analysis/rf_sfs_feature_importance.csv")

    fr_sfs = get_sfs_feature_rank(inputs, labels)
    fr_sfs.to_csv("outputs/feature_analysis/sfs_feature_rank.csv")

    fi_shap = get_rf_sfs_shap(inputs, labels)
    fi_shap.to_csv("outputs/feature_analysis/rf_sfs_shap.csv")
