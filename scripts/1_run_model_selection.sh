#!/bin/bash

python -m src.classification.run_grid_search
python -m src.classification.run_feature_selection_best_model
python -m src.classification.run_prediction_best_model
python -m src.classification.run_prediction_best_model_sfs
