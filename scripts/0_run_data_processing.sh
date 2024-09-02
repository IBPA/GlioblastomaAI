#!/bin/bash

python -m src.data_processing.generate_train_test_splits
python -m src.data_processing.generate_5_fold_cv_splits
