# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import implementations
import copy

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
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
            
            
def preprocessing(y,tx, train = True, columns_to_keep = None):    
    # split data into 4 different categories
    
    def clean_xi(xi, train, columns_to_keep = None):
        """ Function to clean in depth each categorical dataset """
        x = copy.deepcopy(xi)
        # set the -999 to the median of the column
        x = np.where(x==-999, np.nan, x)
        col_median = np.nanmedian(x, axis=0)
        inds_nan = np.where(np.isnan(x))
        x[inds_nan] = np.take(col_median, inds_nan[1])
        
        if train:
            # remove the nan and even columns
            nan_columns = np.all(np.isnan(x), axis=0)
            #print("nan_columns :: ",nan_columns)
            even_columns = np.all(x == x[0,:], axis = 0)
            #print("even_columns :: ",even_columns)
            columns_to_remove = nan_columns | even_columns
            #print("columns_to_remove :: ",columns_to_remove)
            x = x[:,~columns_to_remove]

            # normalize the dataset
            minmax = implementations.dataset_minmax(x)
            implementations.normalize_dataset(x, minmax)

            # Insert a 1 column for non null intersection
            x = np.insert(x, 0, 1, axis=1)
            return x, columns_to_remove
        else:            
            x = x[:,columns_to_keep]
            
            # normalize the dataset
            minmax = implementations.dataset_minmax(x)
            implementations.normalize_dataset(x, minmax)

            # Insert a 1 column for non null intersection
            x = np.insert(x, 0, 1, axis=1)
            return x
            
    
    if train:
        Indices0 = tx[:,22]==0
        y0, x0 = y[Indices0], tx[np.where(Indices0)]

        Indices1 = tx[:,22]==1
        y1, x1 = y[Indices1], tx[np.where(Indices1)]

        Indices2 = tx[:,22]==2
        y2, x2 = y[Indices2], tx[np.where(Indices2)]

        Indices3 = tx[:,22]==3
        y3, x3 = y[Indices3], tx[np.where(Indices3)]
        
        x0, columns_to_remove0 = clean_xi(x0, train)
        x1, columns_to_remove1 = clean_xi(x1, train)
        x2, columns_to_remove2 = clean_xi(x2, train)
        x3, columns_to_remove3 = clean_xi(x3, train)
        return [(y0,x0),(y1,x1),(y2,x2),(y3,x3), (~columns_to_remove0,~columns_to_remove1,~columns_to_remove2,~columns_to_remove3)]
    else:
        i0, i1, i2, i3 = columns_to_keep
        
        rows0 = tx[:,22]==0
        x0 = tx[np.where(rows0)]
        rows1 = tx[:,22]==1
        x1 = tx[np.where(rows1)]
        rows2 = tx[:,22]==2
        x2 = tx[np.where(rows2)]
        rows3 = tx[:,22]==3
        x3 = tx[np.where(rows3)]
        
        x0 = clean_xi(x0, train, i0)
        x1 = clean_xi(x1, train, i1)
        x2 = clean_xi(x2, train, i2)
        x3 = clean_xi(x3, train, i3)
        return [x0, x1, x2, x3, rows0, rows1, rows2, rows3]
    
    
    
