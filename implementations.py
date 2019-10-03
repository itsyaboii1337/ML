import numpy as np
from proj1_helpers import *
### Preprocessing functions

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    ids = [i for i in range(len(x))]
    train_ids = np.random.choice(ids,int(ratio*len(ids)))
    test_ids = []
    for i in range(len(ids)):
        if not(i in train_ids):
            test_ids.append(i)
    
    return x[train_ids],y[train_ids],x[test_ids],y[test_ids]


### Loss functions
def compute_loss(y, tx, w):
    
    N = y.shape[0]
    e = y - np.dot(tx, w)
    L = (1/(2*N))*np.dot(e.transpose(), e)    
    return L

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_loss(y, tx, w))

def compute_gradient(y, tx, w):
    
    N = y.size
    
    return - np.matmul(np.transpose(tx), y-np.matmul(tx, w))/N

def compute_st_gradient(y_n, tx_n, w):
    """ We don't use batches here! """
    return -np.dot(tx_n.transpose(), y_n - np.matmul(tx_n, w))

### Regression functions
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    ws = initial_w
    N = y.shape[0]
    
    for i in range(max_iters):
        
        grad = compute_gradient(y, tx, ws)
        ws = ws - gamma*grad
    
    loss = compute_loss(y, tx, ws)
   
    return loss, ws

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    
    ws = initial_w
    N = y.shape[0]
    
    for i in range (max_iters):
            for n in range (N):
                grad = compute_st_gradient(y[n], tx[n], ws)
                ws = ws - gamma*grad
    
    loss = compute_loss(y, tx, ws)
    
    return loss, ws

def least_squares(y, tx):
    
    Gram = np.dot(np.transpose(tx), tx)
    
    ws = np.dot(np.linalg.inv(Gram), np.dot(np.transpose(tx), y))
    
    loss = compute_loss(y, tx, ws)
    
    return loss, ws

def least_squares_rmse(y, tx):
    
    Gram = np.dot(np.transpose(tx), tx)
    
    ws = np.dot(np.linalg.inv(Gram), np.dot(np.transpose(tx), y))
    
    loss = compute_rmse(y, tx, ws)
    
    return loss, ws

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    Gram = np.dot(np.transpose(tx), tx)
    reg_matrix = lambda_/(2*Gram.shape[0])*np.identity(Gram.shape[0])
    ws = np.dot(np.linalg.inv(Gram+reg_matrix, np.dot(np.transpose(tx), y)))
    loss = compute_rmse(y, tx, ws)    
    return loss, ws

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = [[0]*(degree+1) for n in range(len(x))]
    for n in range(len(x)):
        for i in range(degree+1):
            phi[n][i] = x[n]**i
    return np.array(phi)


def polynomial_regression(y,x,degrees):
    """Constructing the polynomial basis function expansion of the data,
       and then running least squares regression."""
    # define parameters
    # degrees = [1, 3, 7, 12]

    for ind, degree in enumerate(degrees):
        phi_x = build_poly(x,degree)
        loss, ws = least_squares_rmse(y,phi_x)
    return loss, ws
                
### Metrics : accuracy, f1_score
#https://en.wikipedia.org/wiki/Precision_and_recall
def metrics(weights, y_test, x_test):
    tp, fp, tn, fn = 0,0,0,0
    y_pred = predict_labels(weights, x_test)
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
    
    return accuracy, f1_score
