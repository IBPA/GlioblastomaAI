from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils import load_dataset


if __name__ == '__main__':
    inputs_train, labels_train = load_dataset(train=True)
    inputs_test, labels_test = load_dataset(train=False)
    inputs = pd.concat([inputs_train, inputs_test])
    labels = pd.concat([labels_train, labels_test])
    patient_ids = pd.Series(
        [x.split('_')[1] for x in inputs.index.tolist()],
        index=inputs.index,
    )

    # PCA.
    pca = PCA(
        n_components=1,
        random_state=42,
    )
    pca.fit(StandardScaler().fit_transform(inputs))
    pca_result = pd.Series(
        pca.components_[0],
        index=inputs.columns
    ).sort_values(key=abs, ascending=False)
    g = sns.barplot(
        x=pca_result.values[:20],
        y=pca_result.index[:20],
        hue=[0 if x > 0 else 1 for x in pca_result.values[:20]],
        orient='h',
    )
    g.set_title(
        f"Explained variance ratio: {pca.explained_variance_ratio_[0]}%"
    )
    plt.savefig("outputs/feature_extraction/pca.svg")
    plt.close()

    # LDA.
    lda = LinearDiscriminantAnalysis(
        n_components=1,
    )
    lda.fit(inputs, labels)
    lda_result = pd.Series(
        lda.coef_[0],
        index=inputs.columns
    ).sort_values(key=abs, ascending=False)
    g = sns.barplot(
        x=lda_result.values[:20],
        y=lda_result.index[:20],
        hue=[0 if x > 0 else 1 for x in lda_result.values[:20]],
        orient='h',
    )
    g.set_title(
        f"Explained variance ratio: {lda.explained_variance_ratio_[0]}%"
    )
    plt.savefig("outputs/feature_extraction/lda.svg")
    plt.close()
