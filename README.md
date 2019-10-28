# EPFL Machine Learning Higgs 2019
EPFL: CS-433 Machine Learning - Project 1 

**`Team Name`**: **The_Artificially_Intelligent**

**`Team Members`:** Nathan Sennesael, Johan Barthas, Ruben Janssens

## Requirements

* ```Python 3.6``` 
* ```NumPy>=1.15``` 

~~~~shell
cd script
python run.py
~~~~

## Modules and Notebook

### ```proj1_helpers.py```
Helper functions to load raw csv data into NumPy array, generate class predictions and create an output file in csv format for submission to Kaggle.

### ```helpers.py```

Helper functions for data preprocessing, feature engineering and regression model training.

- `preprocessing`: Preprocess train/test data with methods below.
  - `standardize`: Standardize data set ignoring NaN.
  - `delta_angle_norm`, `add_phi`: Add new phi features according to organizer's suggestions.
  - `apply_log1p`: Apply log normalization to long-tailed features.
  - `drop_useless`: Drop useless columns, including raw phi angles, columns with the same values, columns full of NaN.
  - `fill_missing`, `fill_nan`: Mark missing values with NaN and then fill them with zero.
- `train_predict`: Train and predict each group using polynomial ridge regression.
  - `get_jet_index`: Get the index of three groups.
  - `build_poly_feature`: Build polynomial features for input data.

### ```implementations.py```

Six basic machine learning methods with some supported functions.

- `least_squares_GD`: Linear regression using gradient descent.
- `least_squares_SGD`: Linear regression using stochastic gradient descent.
- `least_squares`: Least squares regression using normal equations.
- `ridge_regression`: Ridge regression using normal equations.
- `logistic_regression`: Logistic regression using SGD.
- `reg_logistic_regression`: Regularized logistic regression using SGD.

### ```Cross Validation.ipynb```

Notebook with the cross validation results and related functions.

### ```run.py```

Script to generate the same submission file as we submitted in Kaggle.



