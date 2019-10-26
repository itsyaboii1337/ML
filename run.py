import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy
from implementations import *
from proj1_helpers import *
## Loading the training data
DATA_TRAIN_PATH = 'train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

## Split the dataset
ratio_train_test = 0.9
x_train, y_train, x_test, y_test = split_data(tX, y, ratio_train_test, seed=1)

## Preprocessing training dataset
categorical_datasets = preprocessing(copy.deepcopy(y_train),copy.deepcopy(x_train))
y0, x0 = categorical_datasets[0]
y1, x1 = categorical_datasets[1]
y2, x2 = categorical_datasets[2]
columns_to_keep = categorical_datasets[3]

# clean high correlation areas
#x0
x0 = np.delete(x0, 6, 1)
x0 = np.delete(x0, 6, 1)
#x1
x1 = np.delete(x1, 4, 1)
x1 = np.delete(x1, 18, 1)
x1 = np.delete(x1, 6, 1)
x1 = np.delete(x1, 3, 1)
#x2
x2 = np.delete(x2, 10, 1)
x2 = np.delete(x2, 21, 1)
x2 = np.delete(x2, 4, 1)
x2 = np.delete(x2, 6, 1)
x2 = np.delete(x2, 3, 1)
x2 = np.delete(x2, 4, 1)
x2 = np.delete(x2, 18, 1)

## Hyperparameters found with two parameters grid search (degrees, lambdas)
degree0, lambda0 = 5, 18374.43724610725
degree1, lambda1 = 11, 22.621169304690216
degree2, lambda2 = 10, 1240.6391590588294

## Proceed to data augmentation
phi_x0 = build_poly(x0, degree0)
phi_x1 = build_poly(x1, degree1)
phi_x2 = build_poly(x2, degree2)

## Compute the weights with ridge regression
w0 = ridge_regression(y0,phi_x0,lambda0)[1]
w1 = ridge_regression(y1,phi_x1,lambda1)[1]
w2 = ridge_regression(y2,phi_x2,lambda2)[1]

## Load the test data
DATA_TEST_PATH = 'test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Preprocessing test dataset
categorical_datasets_test = preprocessing(_,copy.deepcopy(tX_test), train = False, columns_to_keep = columns_to_keep)
x0_validate = categorical_datasets_test[0]
x1_validate = categorical_datasets_test[1]
x2_validate = categorical_datasets_test[2]
rows_to_keep_validate0 = categorical_datasets_test[3]
rows_to_keep_validate1 = categorical_datasets_test[4]
rows_to_keep_validate2 = categorical_datasets_test[5]

# clean high correlation areas
#x0
x0_validate = np.delete(x0_validate, 6, 1)
x0_validate = np.delete(x0_validate, 6, 1)
#x1
x1_validate = np.delete(x1_validate, 4, 1)
x1_validate = np.delete(x1_validate, 18, 1)
x1_validate = np.delete(x1_validate, 6, 1)
x1_validate = np.delete(x1_validate, 3, 1)
#x2
x2_validate = np.delete(x2_validate, 10, 1)
x2_validate = np.delete(x2_validate, 21, 1)
x2_validate = np.delete(x2_validate, 4, 1)
x2_validate = np.delete(x2_validate, 6, 1)
x2_validate = np.delete(x2_validate, 3, 1)
x2_validate = np.delete(x2_validate, 4, 1)
x2_validate = np.delete(x2_validate, 18, 1)

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
y_pred[np.where(rows_to_keep_validate0)] = y_pred0
y_pred[np.where(rows_to_keep_validate1)] = y_pred1
y_pred[np.where(rows_to_keep_validate2)] = y_pred2

## Export csv submission file
time_day = datetime.datetime.now().day
time_hour = datetime.datetime.now().hour
time_min = datetime.datetime.now().minute
time_second = datetime.datetime.now().second
time = str(time_day)+"-"+str(time_hour)+"-"+str(time_min)+"-"+str(time_second)
OUTPUT_PATH = 'submission'+"_"+str(time)+".csv"
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)