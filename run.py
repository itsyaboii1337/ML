import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy
from implementations import *
from proj1_helpers import *

# RUN ID : 23779
## Loading the training data
DATA_TRAIN_PATH = 'train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Preprocessing the dataset, log(1+x) of heavy tailed columns, augmentation by adding ones.
x = copy.deepcopy(tX)
x = np.where(x==-999, np.nan, x)
cols = [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]
x[:, cols] = np.log1p(x[:, cols])

# Split the dataset accordingly to the number of jets, we group 2 and 3 for repartition issues
rows0 = x[:,22]==0
rows1 = x[:,22]==1
rows2 = np.logical_or(x[:,22]==2, x[:,22]==3)

y0, x0 = y[rows0], x[np.where(rows0)]
y1, x1 = y[rows1], x[np.where(rows1)]
y2, x2 = y[rows2], x[np.where(rows2)]

# Delete the columns that are phi, nan columns, or constant columns
x0 = np.delete(x0, [4, 5, 6, 12, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29], axis = 1)
x1 = np.delete(x1, [4, 5, 6, 12, 15, 18, 20, 22, 25, 26, 27, 28], axis = 1)
x2 = np.delete(x2, [15, 18, 20, 22, 28], axis = 1)

# Set nanvalues to median value in first column
x0[:,0] = set_median(x0[:,0])
x1[:,0] = set_median(x1[:,0])
x2[:,0] = set_median(x2[:,0])

# Standardize the data
x0, mean_x0, std_x0 = standardize(x0)
x1, mean_x1, std_x1 = standardize(x1)
x2, mean_x2, std_x2 = standardize(x2)

# Add a 1 column
x0 = np.insert(x0, 0, 1, axis=1)
x1 = np.insert(x1, 0, 1, axis=1)
x2 = np.insert(x2, 0, 1, axis=1)

# Split the dataset
ratio = 0.90
x0, y0, _, _ = split_data(x0, y0, ratio, seed=1)
x1, y1, _, _ = split_data(x1, y1, ratio, seed=1)
x2, y2, _, _ = split_data(x2, y2, ratio, seed=1)

## Hyperparameters found with two parameters grid search (degrees, lambdas)
degree0, lambda0 = 11, 94.3604310147891
degree1, lambda1 = 11, 55962.76445319564
degree2, lambda2 = 14, 0.004887374631624427

## Proceed to data augmentation
phi_x0 = build_poly(x0, degree0)
phi_x1 = build_poly(x1, degree1)
phi_x2 = build_poly(x2, degree2)

## Compute the weights with ridge regression
w0 = ridge_regression(y0,phi_x0,lambda0)[0]
w1 = ridge_regression(y1,phi_x1,lambda1)[0]
w2 = ridge_regression(y2,phi_x2,lambda2)[0]

## Load the test data
DATA_TEST_PATH = 'test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Preprocessing the test dataset, log(1+x) of heavy tailed columns, augmentation by adding ones.
cols = [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]
x_validate = np.copy(tX_test)
x_validate = np.where(x_validate==-999, np.nan, x_validate)
x_validate = np.copy(tX_test)
x_validate[:, cols] = np.log1p(x_validate[:, cols])

rows0_validate = x_validate[:,22]==0
rows1_validate = x_validate[:,22]==1
rows2_validate = np.logical_or(x_validate[:,22]==2, x_validate[:,22]==3)
x0_validate = x_validate[np.where(rows0_validate)]
x1_validate = x_validate[np.where(rows1_validate)]
x2_validate = x_validate[np.where(rows2_validate)]

x0_validate = np.delete(x0_validate, [4, 5, 6, 12, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29], axis = 1)
x1_validate = np.delete(x1_validate, [4, 5, 6, 12, 15, 18, 20, 22, 25, 26, 27, 28], axis = 1)
x2_validate = np.delete(x2_validate, [15, 18, 20, 22, 28], axis = 1)

x0_validate[:,0] = set_median(x0_validate[:,0])
x1_validate[:,0] = set_median(x1_validate[:,0])
x2_validate[:,0] = set_median(x2_validate[:,0])

x0_validate = (x0_validate - mean_x0)/(std_x0)
x1_validate = (x1_validate - mean_x1)/(std_x1)
x2_validate = (x2_validate - mean_x2)/(std_x2)

x0_validate = np.insert(x0_validate, 0, 1, axis=1)
x1_validate = np.insert(x1_validate, 0, 1, axis=1)
x2_validate = np.insert(x2_validate, 0, 1, axis=1)

## Proceed to data augmentation
phi_x0_validate = build_poly(copy.deepcopy(x0_validate), degree0)
phi_x1_validate = build_poly(copy.deepcopy(x1_validate), degree1)
phi_x2_validate = build_poly(copy.deepcopy(x2_validate), degree2)

## Predict outcomes
y_pred0 = predict_labels(w0, phi_x0_validate)
y_pred1 = predict_labels(w1, phi_x1_validate)
y_pred2 = predict_labels(w2, phi_x2_validate)

## Merge different datasets
total_length = y_pred0.shape[0] + y_pred1.shape[0] + y_pred2.shape[0]
y_pred = np.zeros((total_length,))
y_pred[np.where(rows0_validate)] = y_pred0
y_pred[np.where(rows1_validate)] = y_pred1
y_pred[np.where(rows2_validate)] = y_pred2

## Export csv submission file
time_day = datetime.datetime.now().day
time_hour = datetime.datetime.now().hour
time_min = datetime.datetime.now().minute
time_second = datetime.datetime.now().second
time = str(time_day)+"-"+str(time_hour)+"-"+str(time_min)+"-"+str(time_second)
OUTPUT_PATH = 'submission'+"_"+str(time)+".csv"
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)