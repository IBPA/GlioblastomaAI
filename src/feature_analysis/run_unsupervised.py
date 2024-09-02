import os

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

from ..utils import load_dataset


def get_pca_component_weight(inputs, labels=None):
    """Run PCA with one component to rank features based on variance.

    Args:
        inputs (pd.DataFrame): Input data.
        labels (pd.Series): Not used.

    Returns:
        pd.DataFrame: Feature importance.

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
    # print(model['pca'].explained_variance_ratio_)

    return pca_result


if __name__ == '__main__':
    os.makedirs("outputs/feature_analysis", exist_ok=True)

    inputs_train, labels_train = load_dataset(train=True)
    inputs_test, labels_test = load_dataset(train=False)
    inputs = pd.concat([inputs_train, inputs_test])
    labels = pd.concat([labels_train, labels_test])

    fi_pca = get_pca_component_weight(inputs, labels)
    fi_pca.to_csv("outputs/feature_analysis/pca_comp_weight.csv")
    # Explained variance ratio: 0.10504044.
