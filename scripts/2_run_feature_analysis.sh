#!/bin/bash

python -m src.feature_analysis.run_supervised
python -m src.feature_analysis.run_unsupervised
python -m src.feature_importance.run
