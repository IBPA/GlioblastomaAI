import os

import numpy as np
import pandas as pd
from plotly.colors import n_colors
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


def plot_features_top_10(path_save=None):
    PATH = "outputs/feature_analysis"

    data_pca = pd.read_csv(f"{PATH}/pca_comp_weight.csv", index_col=0)
    data_rf = pd.read_csv(f"{PATH}/rf_feature_importance.csv", index_col=0)
    data_sfs = pd.read_csv(f"{PATH}/sfs_feature_rank.csv", index_col=0)
    data_lda = pd.read_csv(f"{PATH}/lda_comp_weight.csv", index_col=0)

    data = pd.DataFrame([], index=data_pca.index)
    data['PCA'] = data_pca['rank'].astype(int)
    data['LDA'] = data_lda['rank'].astype(int)
    data['RF'] = data_rf['rank'].astype(int)
    data['SFS'] = data_sfs['rank'].astype(int)

    data['is_common_top_10'] = data.apply(
        lambda row: (row <= 10).sum() > 1,
        axis=1,
    )
    data['is_common_top_20'] = data.apply(
        lambda row: (row <= 20).sum() > 1,
        axis=1,
    )

    data_top_10 = data.query("is_common_top_10 == True")
    data_top_10 = data_top_10.reset_index()
    data_top_10 = data_top_10.rename(columns={'index': 'Biogenic Amine'})
    data_top_10['Biogenic Amine'] = data_top_10['Biogenic Amine'].str.strip()
    column_names = ['Biogenic Amine', 'PCA', 'LDA', 'RF', 'SFS']

    colors = np.array(n_colors(
        'rgb(0, 200, 0)', 'rgb(200, 255, 200)', 10, colortype='rgb'
    ))
    cell_values = [data_top_10[col] for col in column_names]
    cell_colors = [['lightblue'] * len(data_top_10)]
    for vals in cell_values[1:]:
        cell_colors.append([
            colors[val - 1] if val <= 10 else 'white' for val in vals
        ])
    fig = go.Figure(data=[
        go.Table(
            columnwidth=[2, 1, 1, 1, 1],
            header={
                'values': [
                    f"<b>{x}</b>"
                    for x in ['Biogenic<br>Amine', 'PCA', 'LDA', 'RF', 'SFS']
                ],
                'line_color': 'lightblue',
                'fill_color': 'lightblue',
            },
            cells={
                'values': cell_values,
                'line_color': cell_colors,
                'fill_color': cell_colors,
            },
        )
    ])
    fig.write_image(path_save)


def plot_feature_importance_clf(path_save=None):
    PATH = "outputs/feature_analysis"

    data_shap = pd.read_csv(f"{PATH}/rf_sfs_shap.csv", index_col=0)
    data_shap = data_shap.sort_values(by='rank', ascending=False)
    data_shap = data_shap.rename(columns={
        'val_cls0': 'Pre-surgery',
        'val_cls1': 'Post-surgery',
        'val_cls2': 'Pre-radiation',
        'val_cls3': 'Post-radiation',
    })
    data_shap = data_shap.drop(columns=['val', 'rank'])
    data_shap.index = data_shap.index.str.strip()

    sns.set_style("white")
    data_shap.plot(kind='barh', stacked=True, figsize=(10, 10), fontsize=12)
    plt.savefig(path_save)


if __name__ == "__main__":
    os.makedirs("outputs/visualization/feature_analysis", exist_ok=True)

    plot_features_top_10(
        "outputs/visualization/feature_analysis/top_10.svg"
    )
    plot_feature_importance_clf(
        "outputs/visualization/feature_analysis/shap.svg"
    )
