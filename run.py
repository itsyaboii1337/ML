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
(y0,x0,mean_x0,std_x0,y1,x1,mean_x1,std_x1,y2,x2,mean_x2,std_x2, _, _, _) = preprocessing(y, tX, train=True)

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
(x0_validate, x1_validate, x2_validate, rows0_validate, rows1_validate, rows2_validate) = preprocessing(_, tX_test, False, mean_x0, std_x0, mean_x1, std_x1, mean_x2, std_x2)

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