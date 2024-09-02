import os

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils import load_dataset

HUE_ORDER = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
    '15', '16', '17', '18', '20', '22', '23', '24', '25', '26', '27', '28',
    '29', '31', '32', '33', '34', '35', '36', '37', '38', '39'
]
PATIENT_ID_TO_COLOR = {
    patient_id: plt.get_cmap('rainbow')(i / len(HUE_ORDER))
    for i, patient_id in enumerate(HUE_ORDER)
}


def plot_pca(inputs, labels, annotate=False, path_save_dir=None):
    sns.set_style('white')
    plt.figure(figsize=(6, 5))

    X_pca = PCA(
        n_components=2,
        random_state=42,
    ).fit_transform(StandardScaler().fit_transform(inputs.values))

    if annotate:
        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            alpha=0,
        )
        for i, txt in enumerate(labels.tolist()):
            plt.annotate(
                txt,
                (X_pca[i, 0], X_pca[i, 1]),
                ha='center',
                va='center',
                fontsize=12,
                color=PATIENT_ID_TO_COLOR[txt],
            )
    else:
        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            hue=labels.tolist(),
            hue_order=sorted(labels.tolist(), key=int),
        )

    if path_save_dir is not None:
        plt.savefig(f"{path_save_dir}/data_2d_pca.svg")
    plt.close()


def plot_tsne(inputs, labels, annotate=False, path_save_dir=None):
    sns.set_style('white')
    plt.figure(figsize=(6, 5))

    for perp in [5, 10, 20]:
        plt.figure(figsize=(6, 5))
        X_tsne = TSNE(
                n_components=2,
                perplexity=perp,
                random_state=42,
            ).fit_transform(StandardScaler().fit_transform(inputs))

        if annotate:
            sns.scatterplot(
                x=X_tsne[:, 0],
                y=X_tsne[:, 1],
                alpha=0,
            )
            for i, txt in enumerate(labels.tolist()):
                plt.annotate(
                    txt,
                    (X_tsne[i, 0], X_tsne[i, 1]),
                    ha='center',
                    va='center',
                    fontsize=12,
                    color=PATIENT_ID_TO_COLOR[txt],
                )
        else:
            sns.scatterplot(
                x=X_tsne[:, 0],
                y=X_tsne[:, 1],
                hue=labels.tolist(),
                hue_order=sorted(labels.tolist(), key=int),
            )

        if path_save_dir is not None:
            plt.savefig(f"{path_save_dir}/data_2d_tsne_perp{perp}.svg")
        plt.close()


def plot_umap(inputs, labels, annotate=False, path_save_dir=None):
    sns.set_style('white')

    for n_neighbors in [5, 10, 20]:
        plt.figure(figsize=(6, 5))
        X_umap = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric='correlation',
                random_state=42,
                transform_seed=43,
            ).fit_transform(StandardScaler().fit_transform(inputs))

        if annotate:
            sns.scatterplot(
                x=X_umap[:, 0],
                y=X_umap[:, 1],
                alpha=0,
            )
            for i, txt in enumerate(labels.tolist()):
                plt.annotate(
                    txt,
                    (X_umap[i, 0], X_umap[i, 1]),
                    ha='center',
                    va='center',
                    fontsize=12,
                    color=PATIENT_ID_TO_COLOR[txt],
                )
        else:
            sns.scatterplot(
                x=X_umap[:, 0],
                y=X_umap[:, 1],
                hue=labels.tolist(),
                hue_order=sorted(labels.tolist(), key=int),
            )

        if path_save_dir is not None:
            plt.savefig(f"{path_save_dir}/data_2d_umap_nn{n_neighbors}.svg")
        plt.close()


if __name__ == '__main__':
    inputs_train, labels_train = load_dataset(train=True)
    inputs_test, labels_test = load_dataset(train=False)

    inputs = pd.concat([inputs_train, inputs_test])
    labels = pd.concat([labels_train, labels_test])
    patient_ids = pd.Series(
        [x.split('_')[1] for x in inputs.index.tolist()],
        index=inputs.index,
    )

    # Plot all samples.
    path_save_dir = "outputs/visualization/data_2d/treatment"
    os.makedirs(path_save_dir, exist_ok=True)
    plot_pca(inputs, labels, path_save_dir=path_save_dir)
    plot_tsne(inputs, labels, path_save_dir=path_save_dir)
    plot_umap(inputs, labels, path_save_dir=path_save_dir)

    path_save_dir = "outputs/visualization/data_2d/patient/all"
    os.makedirs(path_save_dir, exist_ok=True)
    plot_pca(inputs, patient_ids, annotate=True, path_save_dir=path_save_dir)
    plot_tsne(inputs, patient_ids, annotate=True, path_save_dir=path_save_dir)
    plot_umap(inputs, patient_ids, annotate=True, path_save_dir=path_save_dir)

    # Plot patient subsets by each stage.
    for label in labels.unique():
        path_save_dir = f"outputs/visualization/data_2d/patient/{label}"
        os.makedirs(path_save_dir, exist_ok=True)
        inputs_ = inputs[labels == label]
        labels_ = patient_ids[labels == label]
        plot_pca(inputs_, labels_, annotate=True, path_save_dir=path_save_dir)
        plot_tsne(inputs_, labels_, annotate=True, path_save_dir=path_save_dir)
        plot_umap(inputs_, labels_, annotate=True, path_save_dir=path_save_dir)

    patient_id_label_df = pd.concat([patient_ids, labels], axis=1)
    patient_id_label_df.columns = ['patient_id', 'label']
    patient_id_label_map = patient_id_label_df.groupby('patient_id').apply(
        lambda group: pd.Series(
            {
                f'label_{i}': i in group['label'].value_counts().index.tolist()
                for i in range(4)
            }
        )
    )

    def _get_input(group):
        inputs_ = pd.DataFrame(
            inputs.loc[group['sample_id']].values.reshape(1, -1)[0]
        ).T
        inputs_.columns = [
            f"{j}_{i}"
            for i in range(len(group))
            for j in inputs.columns
        ]

        return inputs_

    # Plot patient subsets by grouping stages.
    for label_set in [[0, 1], [0, 1, 2]]:
        path_save_dir \
            = f"outputs/visualization/data_2d/patient/" \
            f"{'+'.join([str(x) for x in label_set])}"
        os.makedirs(path_save_dir, exist_ok=True)

        query_str = ' & '.join(
            ['(label_' + str(label) + ' == True)' for label in label_set]
        )

        patients = patient_id_label_map.query(query_str).index.tolist()
        patient_id_label_df_ = patient_id_label_df.query(
            f"patient_id in {patients} & label in {label_set}"
        ).sort_values(['patient_id', 'label']).reset_index()

        inputs_ = patient_id_label_df_.groupby('patient_id').apply(_get_input)
        inputs_.index = inputs_.index.droplevel(1)
        labels_ = pd.Series(inputs_.index, index=inputs_.index)

        plot_pca(inputs_, labels_, annotate=True, path_save_dir=path_save_dir)
        plot_tsne(inputs_, labels_, annotate=True, path_save_dir=path_save_dir)
        plot_umap(inputs_, labels_, annotate=True, path_save_dir=path_save_dir)
