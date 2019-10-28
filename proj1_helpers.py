# -*- coding: utf-8 -*-
import csv
import numpy as np
import implementations
import copy

""" Loading the data tool """

def load_csv_data(data_path, sub_sample=False):
    """
    Loads data and returns y (class labels), tX (features) and ids (event ids)
    Arguments:
    - data_path: the path of the csv file
    - sub_sample: whether we should sub_sample or not, default: False
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

            
""" Preprocessing tools""" 

def split_data(x, y, ratio=0.8, seed=1):
    """
    Split the dataset based on the split ratio. If ratio is 0.8 you will have 80% of your data set dedicated to training and the rest dedicated to testing.
    Arguments:
    - y: the column of ground truth results
    - x: the features matrix
    - ratio: the split ratio between 0 and 1, default: 0.8
    - seed: the seed used during the pseudo-random shuffle, use the same seed for the same output
    """
    # Set the seed
    np.random.seed(seed)
    ids = np.random.rand(len(y))
    train_ids = ids<0.9
    test_ids = ids>0.9

    return x[train_ids], y[train_ids], x[test_ids], y[test_ids]

def set_median(x):
    median = np.nanmedian(x, axis=0)
    inds_nan = np.isnan(x)
    x[inds_nan] = median
    return x

def standardize(x):
    """
    Standardize the original data set x. You should make a deepcopy before computing this function.
    Argument : x: the features matrix
    """
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x
    
def preprocessing(y, x, train=True,mean_x0=[], std_x0=[], mean_x1=[], std_x1=[], mean_x2=[], std_x2=[]):
    """
    This method cleans the data, split the data accordingly to the number of jets,
    select the features for each category and standardize them.
    Arguments:
    - y: the column of ground truth results
    - x: the features matrix
    - train: True if we preprocess a dataset for training, default: True
    """
    # Preprocessing the dataset, log(1+x) of heavy tailed columns, augmentation by adding ones.
    x = copy.deepcopy(x)
    x = np.where(x==-999, np.nan, x)
    cols = [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]
    x[:, cols] = np.log1p(x[:, cols])

    # Split the dataset accordingly to the number of jets, we group 2 and 3 for repartition issues
    rows0 = x[:,22]==0
    rows1 = x[:,22]==1
    rows2 = np.logical_or(x[:,22]==2, x[:,22]==3)
    
    x0 = x[np.where(rows0)]
    x1 = x[np.where(rows1)]
    x2 = x[np.where(rows2)]
    if train:
        y0 = y[rows0]
        y1 = y[rows1]
        y2 = y[rows2]
    
    def cleanup_xi(feat_matrix, columns_to_delete=[], train=True, mean_xi=[], std_xi=[]):
        """
        This method remove the columns to delete, set nanvalues to median value in first column,
        standardize the data and add 1 column to the features matrix.
        Arguments:
        - feat_matrix: features matrix
        - columns_to_delete: columns to be deleted, default=[]
        - mean_xi: mean of each columns for the category i (to be used for testing dataset), default=[]
        - std_xi: standard deviation of each columns for the category i (to be used for testing dataset), default=[]
        """
        xi = copy.deepcopy(feat_matrix)
        xi = np.delete(xi, columns_to_delete, axis = 1)
        xi[:,0] = set_median(xi[:,0])
        if train:
            xi, mean_xi, std_xi = standardize(xi)
            xi = np.insert(xi, 0, 1, axis=1)
            return xi, mean_xi, std_xi
        xi = (xi-mean_xi)/std_xi
        xi = np.insert(xi, 0, 1, axis=1)
        return xi

    if train:
        x0, mean_x0, std_x0 = cleanup_xi(x0, [4, 5, 6, 12, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29], True)    
        x1, mean_x1, std_x1 = cleanup_xi(x1, [4, 5, 6, 12, 15, 18, 20, 22, 25, 26, 27, 28], True) 
        x2, mean_x2, std_x2 = cleanup_xi(x2, [15, 18, 20, 22, 28], True)

        # Split the dataset with 90%
        ratio = 0.90
        x0, y0, x0_test, y0_test = split_data(x0, y0, ratio, seed=1)
        x1, y1, x1_test, y1_test = split_data(x1, y1, ratio, seed=1)
        x2, y2, x2_test, y2_test = split_data(x2, y2, ratio, seed=1)
        return (y0,x0,mean_x0,std_x0,y1,x1,mean_x1,std_x1,y2,x2,mean_x2,std_x2, rows0, rows1, rows2)

    x0 = cleanup_xi(x0, [4, 5, 6, 12, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29], False, mean_x0, std_x0)    
    x1 = cleanup_xi(x1, [4, 5, 6, 12, 15, 18, 20, 22, 25, 26, 27, 28], False, mean_x1, std_x1) 
    x2 = cleanup_xi(x2, [15, 18, 20, 22, 28], False, mean_x2, std_x2)
    return (x0, x1, x2, rows0, rows1, rows2)
    


def build_poly(x, degree):
    """
    Build polynomial features for input data x with the degree.
    Arguments:
    - x: features matrix
    - degree: the degree of the polynom used for data augmentation
    """
    phi_x = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        phi_x = np.c_[phi_x, np.power(x, deg)]
    # Square root of absolute value
    phi_x = np.c_[phi_x, np.sqrt(np.abs(x))]
    # Vectorizing the computing
    i, j = np.triu_indices(x.shape[1], 1)
    phi_x = np.c_[phi_x, x[:, i] * x[:, j]]
    return phi_x




""" Prediction tools """

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred

def predict_labels_log_reg(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = implementations.sigmoid_activation(np.dot(data, weights))
    y_pred[np.where(y_pred < 0.5)] = -1
    y_pred[np.where(y_pred >= 0.5)] = 1
    return y_pred

""" Metrics tools """
#Ref: https://en.wikipedia.org/wiki/Precision_and_recall
def metrics(y_test, y_pred):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if (y_pred[i] == 1):
            if (y_test[i] == 1):
                tp += 1
            else:
                fp += 1
        else:
            if (y_test[i] == 1):
                fn += 1
            else:
                tn += 1
    #precision = tp/(tp+fp)
    #recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    f1_score = 2*tp/(2*tp+tn+fp+fn)

    return accuracy, f1_score#, precision, recall

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_fold, lambda_, degree):
    """
    Return the loss for ridge regression for this given lambda_ and given degree
    Arguments:
    - y: the column of ground truth results
    - x: the features matrix
    - k_fold: the number of fold to do cross validation
    - lambda_: penalizing parameter for ridge regression
    - degree: the degree of the polynomial data augmentation
    """
    k_indices = build_k_indices(y, k_fold, 1)
    x_k, y_k = x[k_indices], y[k_indices]
    Loss_tr = []
    Loss_te = []
    for k in range(k_fold): 
        x_train, y_train, x_test, y_test = [],[],[],[]        
        x_test = x_k[k]
        y_test = y_k[k]        
        x_train = np.delete(x_k, k, axis = 0)
        y_train = np.delete(y_k, k, axis = 0)
        phi_x_train = build_poly(x_train, degree)
        phi_x_test = build_poly(x_test, degree)
        loss_tr, weights = implementations.ridge_regression(y_train, phi_x_train, lambda_)
        loss_te = implementations.compute_mse(y_test, phi_x_test, weights)
        Loss_tr.append(loss_tr)
        Loss_te.append(loss_te)
    Loss_tr = np.array(Loss_tr)
    Loss_te = np.array(Loss_te)
    
    return Loss_tr.mean(), Loss_te.mean()

""" Exporting the results tool """

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w',newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})    
