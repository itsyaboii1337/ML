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
- `predict_labels & predict_labels_log_reg`: Returns prediction based on the weights of the model
- `metrics`: Returns the accuracy and F1 score of a prediction
- `create_csv_submission`: Makes csv submission for given prediction

### ```implementations.py```
- `cross_validation`: Cross validation
- `least_squares_GD`: Linear regression using gradient descent.
- `least_squares_SGD`: Linear regression using stochastic gradient descent.
- `least_squares`: Least squares regression using normal equations.
- `ridge_regression`: Ridge regression using normal equations.
- `logistic_regression`: Logistic regression using SGD.
- `reg_logistic_regression`: Regularized logistic regression using SGD.
- `grid_search_ridge_regression`: Grid search on polynomial degrees and penalizing parameter with ridge regression

Note1: implementations.py also contains many minor functions that are either used within one of the above functions or used within a notebook but not for any final results.

Note2: At the very bottom of implementations.py you find code which was an attempt to build a neural nerwork from scratch. This attempt was succesful in the sense that it gave slightly better results than our final submittion when enough epochs were ran. However, Since the neural network only has 1 weighted layer and 1 bias layer, the improvements were only minor while the run time was significantly longer. Attempts of adding an additional hidden layer were unsuccesful, therefore this method was not mentioned in the final report.

### ```run.py```

Script to generate the same submission file as we submitted to AIcrowd



