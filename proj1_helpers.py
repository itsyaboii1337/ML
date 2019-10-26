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
    

def preprocessing(y,tx, train = True, columns_to_keep = None):    
    """
    Return a cleaned dataset (y,tx), operations on it depends whether it's a training dataset or a testing dataset
    Arguments:
    - y: the column to be predicted, can be None if in testing mode
    - tx: the features matrix
    - train: whether we are in training mode or testing mode, default: training mode
    - columns_to_keep: required if we are in testing mode
    """
    
    def clean_xi(xi, train, columns_to_keep = None):
        """ Function to clean in depth each categorical dataset """
        x = copy.deepcopy(xi)
        # Set the -999 value to the median of the column
        x = np.where(x==-999, np.nan, x)
        col_median = np.nanmedian(x, axis=0)
        inds_nan = np.where(np.isnan(x))
        x[inds_nan] = np.take(col_median, inds_nan[1])
        
        if train:
            # Remove the Nan and constant columns
            nan_columns = np.all(np.isnan(x), axis=0)
            cst_columns = np.all(x == x[0,:], axis = 0)
            columns_to_remove = nan_columns | cst_columns
            x = x[:,~columns_to_remove]
            # Standardize the features         
            x,_,_ = implementations.standardize(x)
            # Insert a 1 column for a non null intersection
            x = np.insert(x, 0, 1, axis=1)
            return x, columns_to_remove
        else:  
            # Remove the same columns that were removed during the preprocessing of the training phase   
            x = x[:,columns_to_keep]
            # Standardize the features  
            x,_,_ = implementations.standardize(x)
            # Insert a 1 column for non null intersection
            x = np.insert(x, 0, 1, axis=1)
            return x

    # Split the rows base on categories of 22th column of features which is the number of jets       
    rows0 = tx[:,22]==0
    rows1 = tx[:,22]==1
    rows2 = np.logical_or(tx[:,22]==2, tx[:,22]==3)

    if train:
        # Split the training dataset 
        y0, x0 = y[rows0], tx[np.where(rows0)]
        y1, x1 = y[rows1], tx[np.where(rows1)]
        y2, x2 = y[rows2], tx[np.where(rows2)]
        # Clean the training dataset with the nested function and collect removed columns
        x0, columns_to_remove0 = clean_xi(x0, train)
        x1, columns_to_remove1 = clean_xi(x1, train)
        x2, columns_to_remove2 = clean_xi(x2, train)
        return [(y0,x0),(y1,x1),(y2,x2), (~columns_to_remove0,~columns_to_remove1,~columns_to_remove2)]
    else:
        # Get back the columns that should be kept in the testing dataset
        i0, i1, i2 = columns_to_keep
        # Split the testing dataset
        x0 = tx[np.where(rows0)]        
        x1 = tx[np.where(rows1)]        
        x2 = tx[np.where(rows2)]
        # Clean the testing dataset with the nested function using the columns that were remove during the training phase
        x0 = clean_xi(x0, train, i0)
        x1 = clean_xi(x1, train, i1)
        x2 = clean_xi(x2, train, i2)
        return [x0, x1, x2, rows0, rows1, rows2]

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
