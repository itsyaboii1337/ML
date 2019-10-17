import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt
# Preprocessing functions


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    ids = [i for i in range(len(x))]
    train_ids = np.random.choice(ids, int(ratio*len(ids)))
    test_ids = []
    for i in range(len(ids)):
        if not(i in train_ids):
            test_ids.append(i)

    return x[train_ids], y[train_ids], x[test_ids], y[test_ids]


def dataset_minmax(tx):
    minmax = list()
    for i in range(len(tx[0])):
        col_values = [row[i] for row in tx]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# calculate column means


def column_means(tx):
    means = [0 for i in range(len(tx[0]))]
    for i in range(len(tx[0])):
        col_values = [row[i] for row in tx]
        means[i] = sum(col_values) / float(len(tx))
    return means

# calculate column standard deviations


def column_stdevs(tx, means):
    stdevs = [0 for i in range(len(tx[0]))]
    for i in range(len(tx[0])):
        variance = [pow(row[i]-means[i], 2) for row in tx]
        stdevs[i] = sum(variance)
    stdevs = [np.sqrt(x/(float(len(tx)-1))) for x in stdevs]
    return stdevs

# standardize dataset


def standardize_dataset(tx, means, stdevs):
    for row in tx:
        for i in range(len(row)):
            if (stdevs[i]!=0):
                row[i] = (row[i] - means[i]) / stdevs[i]

# Rescale dataset columns to the range 0-1


def normalize_dataset(tx, minmax):
    for row in tx:
        for i in range(len(row)):
            if((minmax[i][1] - minmax[i][0])!=0):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
                          
    
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, num_k, lambda_, degree):
    """return the loss of ridge regression."""
    x_k, y_k = x[k_indices], y[k_indices]
    Weights = []
    Train_accs = []
    Test_accs = []

    
    for k in range(num_k):
        first = True
        for i in range(len(k_indices)):
            if (i != k):
                if first:
                    x_train = x_k[i]
                    y_train = y_k[i]
                else:
                    x_train = np.concatenate(x_train, x_k[i])
                    y_train = np.concatenate(y_train, y_k[i])
            else :
                x_test = x_k[i]
                y_test = y_k[i]
                
        phi_x_train = build_poly(x_train, degree)
        phi_x_test = build_poly(x_test, degree)
        loss_tr, weights = ridge_regression(y_train, phi_x_train, lambda_)        
        cat_accuracy_train, f1_score_train = metrics(weights,y_train,phi_x_train)         
        cat_accuracy_test, f1_score_test= metrics(weights,y_test,phi_x_test)
        Weights.append(weights)
        Train_accs.append(cat_accuracy_train)
        Test_accs.append(cat_accuracy_test)
    
    Train_accs = np.array(Train_accs)
    Test_accs = np.array(Test_accs)
    Weights = np.array(Weights)


    
    
    return np.mean(Train_accs), np.mean(Test_accs), np.mean(Weights) 
    
    


# Loss functions


def compute_loss(y, tx, w):

    N = y.shape[0]
    e = y - np.dot(tx, w)
    L = (1/(2*N))*np.dot(e.transpose(), e)
    return L


def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_loss(y, tx, w))

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0/(1+np.exp(-t))

def compute_loss_log_reg(y, tx, w):
    """compute the cost by negative log likelihood."""
    N = y.shape[0]
    loss = 0
    for n in range(N):
        sigmoid_value = sigmoid(np.dot(tx[n,:].T, w))
        loss += -(y[n]*np.log(sigmoid_value)+(1-y[n])*np.log(1-sigmoid_value))
    return loss

# Gradient functions

def compute_gradient(y, tx, w):
    N = y.size
    return -np.dot(np.transpose(tx), y-np.dot(tx, w))/N


def compute_gradient_log_reg(y, tx, w):
    """compute the gradient of loss."""
    sigmoid_value = sigmoid(np.dot(tx,w))
    return np.dot(tx.T,sigmoid_value-y)

# Hessian functions

def compute_hessian_log_reg(y, tx, w):
    """return the hessian of the loss function."""
    N = y.shape[0]
    hessian_diag = np.zeros(N)
    for n in range(N):
        sigmoid_value = sigmoid(np.dot(tx[n,:],w))
        hessian_diag[n] = np.dot(sigmoid_value,1-sigmoid_value)
    S = np.diag(hessian_diag)
    return np.dot(tx.T,np.dot(S,tx))




# Regression functions

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = compute_loss_log_reg(y, tx, w)
    gradient = compute_gradient_log_reg(y, tx, w)
    hessian = compute_hessian_log_reg(y, tx, w)
    return loss, gradient, hessian

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = compute_loss_log_reg(y,tx,w) + lambda_/2*np.dot(w.T,w)
    gradient = compute_gradient_log_reg(y,tx,w) + lambda_*w
    #hessian = compute_hessian_log_reg(y,tx,w) + lambda_
    return loss, gradient#, hessian


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

    for i in range(max_iters):
        for n in range(N):
            grad = compute_gradient(y[n], tx[n], ws)
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
    ws = np.dot(np.linalg.inv(Gram+reg_matrix), np.dot(np.transpose(tx), y))
    loss = compute_rmse(y, tx, ws)
    return loss, ws

def build_poly(x, degree):
    """Build polynomial features for input data x."""
    # 0 to degree
    phi_x = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        phi_x = np.c_[phi_x, np.power(x, deg)]
    # square root of absolute value
    phi_x = np.c_[phi_x, np.sqrt(np.abs(x))]
    # cross terms
    # vectorize the calculation
    # ref: https://stackoverflow.com/questions/22041561/python-all-possible-products-between-columns
    i, j = np.triu_indices(x.shape[1], 1)
    phi_x = np.c_[phi_x, x[:, i] * x[:, j]]
    return phi_x


def polynomial_regression(y, x, degrees):
    """Constructing the polynomial basis function expansion of the data,
       and then running least squares regression."""
    # define parameters
    # degrees = [1, 3, 7, 12]

    for ind, degree in enumerate(degrees):
        phi_x = build_poly(x, degree)
        loss, ws = least_squares_rmse(y, phi_x)
    return loss, ws

# One step gradient descent

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_loss_log_reg(y, tx, w)
    gradient = compute_gradient_log_reg(y, tx, w)
    w = w - gamma*gradient
    return loss, w

def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, gradient, hessian = logistic_regression(y, tx, w)
    w = w - gamma*np.dot(np.linalg.inv(hessian),gradient)
    return loss, w

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient= penalized_logistic_regression(y, tx, w, lambda_)#, hessian 
    w = w - gamma*gradient
    return loss, w



# Metrics : accuracy, f1_score
# https://en.wikipedia.org/wiki/Precision_and_recall
def metrics(weights, y_test, x_test, predict=predict_labels):
    tp, fp, tn, fn = 0, 0, 0, 0
    y_pred = predict(weights, x_test)
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


# PostProcessing functions
def plot_acc_f1(cat_acc, f1_score, lambdas):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, cat_acc, color='b', marker='*',
                 label="Categorical accuracy")
    plt.semilogx(lambdas, f1_score, color='r', marker='*', label="f1 score")
    plt.xlabel("lambda")
    plt.ylabel("Value")
    plt.title("Ridge regression")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("ridge_regression")
