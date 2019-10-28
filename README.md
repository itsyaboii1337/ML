# EPFL Machine Learning Higgs 2019
EPFL: CS-433 Machine Learning - Project 1 

**`Team Name`**: **The_Artificially_Intelligent**

**`Team Members`:** Nathan Sennesael, Johan Barthas, Ruben Janssens

## Requirements

* ```Python 3.6``` 
* ```NumPy 1.17``` 

## Submitted Files

### ```proj1_helpers.py```
- `load_csv_data`: Load csv data into numpy arrays.
- `split_data`: Split data into train and test.
- `set_median`: Used to turn nan values into the median of the corresponding column
- `standardize`: Standardizes a column (zero mean and unit variance)
- `preprocessing`: Preprocessing of the data (log-scale data, categorical splitting, delete highly correlated columns, remove nan columns, replace nan values by column median)
- `build_poly`: Turns numpy array into polynomial augmentation of itself
- `predict_labels & predict_labels_log_reg`: returns prediction based on the weights of the model
- `metrics`: returns the accuracy and F1 score of a prediction
- `create_csv_submission`: makes csv submission for given prediction

### ```implementations.py```
- `build_k_indices`: Build k indices for cross validation
- `cross_validation`: Cross validation
- `compute_mse`: Compute the mean square error of the predicted output compared to the ground truth results.
- `compute_mse`: Compute the root mean square error of the predicted output compared to the ground truth results.
- ``: Linear regression using gradient descent.
- `least_squares_SGD`: Linear regression using stochastic gradient descent.
- `least_squares`: Least squares regression using normal equations.
- `ridge_regression`: Ridge regression using normal equations.
- `logistic_regression`: Logistic regression using SGD.
- `reg_logistic_regression`: Regularized logistic regression using SGD.

### ```Cross Validation.ipynb```

Notebook with the cross validation results and related functions.

### ```run.py```

Script to generate the same submission file as we submitted in Kaggle.



