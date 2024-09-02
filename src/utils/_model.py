import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


def initialize_best_model():
    # Initialize the model with the optimal hparams.
    gscv_results = pd.read_csv(
        "outputs/classification/gscv_results_summary.csv"
    )
    best_model_stat = gscv_results.iloc[0]

    if best_model_stat['classifier'] == 'lr':
        model = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('cls', LogisticRegression()),
            ]
        )
    elif best_model_stat['classifier'] == 'svm':
        model = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('cls', SVC()),
            ]
        )
    elif best_model_stat['classifier'] == 'rf':
        model = RandomForestClassifier()
    elif best_model_stat['classifier'] == 'ada':
        model = AdaBoostClassifier()
    elif best_model_stat['classifier'] == 'mlp':
        model = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('cls', MLPClassifier()),
            ]
        )
    else:
        raise ValueError(
            f"Model {best_model_stat['classifier']} not supported."
        )
    best_hparam = eval(best_model_stat['params'])
    model.set_params(**best_hparam)

    return model
