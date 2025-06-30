# Spaceship Titanic

## Overview
This repository is for creating a submission to the following Kaggle competition:
https://www.kaggle.com/competitions/spaceship-titanic/overview

## Files & Folders
- data_in: Input CSV files
- edazzio_ex.py: Example project to draw knowledge from
- hyperparameter_tuning.py: Takes the best x models from model_comparison.py and finds the optimal parameters
- model_comparison.py: Compares accuracy for a variety of models on the data
- requirements.txt: For use with PIP and virtual environment
- stacking_comparison.py: Stacks tuned models from hyperparameter_tuning.py and finds most accurate combination
- titanic_model.py: Main script. Reads & cleans data, performs feature engineering, makes predictions, outputs submission CSV file
- visualize.py: Used by titanic_model.py if VISUALIZE flag is set to TRUE. Creates helpful visualizations of data
