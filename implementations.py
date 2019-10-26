import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt


def build_k_indices(y, k_num, seed=1):
    """
    Build k indices for k-fold.
    Arguments: 
    - y: the ground truth matrix
    - k_num: the number of folds
    - seed: the seed used during the pseudo-random shuffle, use the same seed for the same output
    """
    num_row = y.shape[0]
    interval = int(num_row / k_num)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_num)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, num_k, lambda_, degree):
    """return the loss of ridge regression."""
    x_k, y_k = x[k_indices], y[k_indices]
    Weights = []
    Train_accs = []
    Test_accs = []

    
    for k in range(num_k):
        first = True
        x_train, y_train, x_test, y_test = [],[],[],[]
        
        x_test = x_k[k]
        y_test = y_k[k]
        
        x_train = np.concatenate(np.delete(x_k, k, axis = 0))
        y_train = np.concatenate(np.delete(y_k, k, axis = 0))
                
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
    
    return np.mean(Train_accs), np.mean(Test_accs), Weights[0]

    
    


# Loss functions

def compute_mse(y, tx, w):
    """
    Compute the mean square error of the predicted output compared to the ground truth results.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - w: weights for the linear model
    """
    N = y.shape[0]
    e = y - np.dot(tx, w)
    L = (1/(2*N))*np.dot(e.transpose(), e)
    return L

def compute_rmse(y, tx, w):
    """
    Compute the root mean square error of the predicted output compared to the ground truth results.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - w: weights for the linear model
    """
    return np.sqrt(2*compute_mse(y, tx, w))

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
    
def sigmoid_deriv(x):
    return x * (1 - x)

def relu(x):
    if x < 0:
        return 0
    else:
        return x
    
    
def compute_loss_log_reg(y, tx, w):
    """
    - y: matrix of ground truth results
    - tx: features matrix
    - w: weights for the model
    """
    eps = 1e-15
    sig = sigmoid(tx.dot(w))
    sig = np.clip(sig, eps, 1-eps)

    loss = y.T.dot(np.log(sig)) + (1 - y).T.dot(np.log(1 - sig))

    return np.squeeze(- loss).item()


def compute_loss_reg_log_reg(y, tx, w, lambda_):
    """
    Compute the cost by negative log likelihood with a ridge term.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - w: weights for the model
    """
    loss = compute_loss_log_reg(y,tx,w) + lambda_/2*np.dot(w.T,w)
    return loss

# Gradient functions

def compute_gradient_mse(y, tx, w):
    """
    Compute gradient in mse context.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - w: weights for the linear model
    """
    N = y.size
    return -np.dot(np.transpose(tx), y-np.dot(tx, w))/N


def compute_gradient_log_reg(y, tx, w):
    """"
    Compute gradient in logistic regression context.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - w: weights for the model
    """
    sigmoid_value = sigmoid(np.dot(tx,w))
    return np.dot(tx.T,sigmoid_value-y)

# Hessian functions

def compute_hessian_log_reg(y, tx, w):
    """
    Compute hessian in logistic regression context.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - w: weights for the model
    """
    N = y.shape[0]
    hessian_diag = np.zeros(N)
    for n in range(N):
        sigmoid_value = sigmoid(np.dot(tx[n,:],w))
        hessian_diag[n] = np.dot(sigmoid_value,1-sigmoid_value)
    S = np.diag(hessian_diag)
    return np.dot(tx.T,np.dot(S,tx))

# All inclusive functions for loss, gradient and hessian

def compute_log_reg(y, tx, w):
    """
    Compute loss, gradient and hessian in logistic regression context.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - w: weights for the model
    """
    loss = compute_loss_log_reg(y, tx, w)
    gradient = compute_gradient_log_reg(y, tx, w)
    hessian = compute_hessian_log_reg(y, tx, w)
    return loss, gradient, hessian

def compute_pen_log_reg(y, tx, w, lambda_):
    """
    Compute loss, gradient and hessian in penalized logistic regression context.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - w: weights for the model
    """
    loss = compute_loss_log_reg(y,tx,w) + lambda_/2*np.dot(w.T,w)
    gradient = compute_gradient_log_reg(y,tx,w) + lambda_*w
    hessian = compute_hessian_log_reg(y,tx,w) + lambda_
    return loss, gradient, hessian


# Regression functions

def least_squares_GD(y, tx, initial_w, max_iters=1000, gamma=0.001):
    """
    Linear regression using gradient descent.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - initial_w: initial weights for the model
    - max_iters: maximum number of iterations during the gradient descent, default: 1000
    - gamma: the learning rate, default: 0.001
    """
    ws = initial_w
    N = y.shape[0]
    for i in range(max_iters):
        grad = compute_gradient_mse(y, tx, ws)
        ws = ws - gamma*grad
    loss = compute_mse(y, tx, ws)
    return loss, ws

def least_squares_SGD(y, tx, initial_w, max_iters=1000, gamma=0.001):
    """
    Linear regression using stochastic gradient descent with mini-batch size 1.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - initial_w: initial weights for the model
    - max_iters: maximum number of iterations during the gradient descent, default: 1000
    - gamma: the learning rate, default: 0.001
    """
    ws = initial_w
    N = y.shape[0]
    for i in range(max_iters):
        #for n in range(N):
        rand_ind = np.random.randint(0, N)
        grad = compute_gradient_mse(y[rand_ind], tx[rand_ind], ws)
        ws = ws - gamma*grad
    loss = compute_mse(y, tx, ws)
    return loss, ws

def least_squares(y, tx):
    """
    Linear regression using normal equations. Be sure to have a non singular matrix.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    """
    Gram = np.dot(np.transpose(tx), tx)
    ws = np.dot(np.linalg.inv(Gram), np.dot(np.transpose(tx), y))
    loss = compute_mse(y, tx, ws)
    return loss, ws

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations. Be sure to have a non singular matrix.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - lambda_: the penalizing parameter
    """
    Gram = np.dot(np.transpose(tx), tx)
    reg_matrix = lambda_/(2*Gram.shape[0])*np.identity(Gram.shape[0])
    ws = np.dot(np.linalg.inv(Gram+reg_matrix), np.dot(np.transpose(tx), y))
    loss = compute_mse(y, tx, ws)
    return loss, ws

def logistic_regression(y, tx, initial_w, max_iters=1000, gamma=0.001):
    """
    Logistic regression using stochastic gradient descent with mini-batch size 1.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - initial_w: initial weights for the model
    - max_iters: maximum number of iterations during the gradient descent, default: 1000
    - gamma: the learning rate, default: 0.001
    """
    ws = initial_w
    N = y.shape[0]
    for i in range(max_iters):
        rand_ind = np.random.randint(0, N)
        grad = compute_gradient_log_reg(y[rand_ind], tx[rand_ind], ws)
        ws = ws - gamma*grad
        loss = compute_loss_log_reg(y, tx, ws)
        if i%(max_iters/10) == 0:
            print("The loss for step {} is {}.".format(i,loss))
    loss = compute_loss_log_reg(y, tx, ws)
    return loss, ws

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters=1000, gamma=0.001):
    """
    Penalized logistic regression using stochastic gradient descent with mini-batch size 1.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - lambda_: the penalizing parameter
    - initial_w: initial weights for the model
    - max_iters: maximum number of iterations during the gradient descent, default: 1000
    - gamma: the learning rate, default: 0.001
    """
    ws = initial_w
    N = y.shape[0]
    for i in range(max_iters):
        rand_ind = np.random.randint(0, N)
        grad = compute_gradient_log_reg(y[rand_ind], tx[rand_ind], ws) + lambda_*ws
        ws = ws - gamma*grad
        loss = compute_loss_reg_log_reg(y, tx, ws, lambda_)
        if i%(max_iters/10) == 0:
            print("The loss for step {} is {}.".format(i, loss))
        
    loss = compute_loss_reg_log_reg(y, tx, ws, lambda_)
    return loss, ws

def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters=1000, gamma=0.001):
    """
    Penalized logistic regression using newton method.
    Arguments:
    - y: matrix of ground truth results
    - tx: features matrix
    - lambda_: the penalizing parameter
    - initial_w: initial weights for the model
    - max_iters: maximum number of iterations during the gradient descent, default: 1000
    - gamma: the learning rate, default: 0.001
    """
    ws = initial_w
    N = y.shape[0]
    for i in range(max_iters):
        loss, gradient, hessian = compute_pen_log_reg(y, tx, ws, lambda_)
        ws = ws - gamma*np.dot(np.linalg.inv(hessian),gradient)
        print("The loss for step {} is {}.".format(i, loss))
    return loss, ws

# Neural Network (1 weighter layer, 1 bias layer, activation layer)
def create_neural_network(layer_array, input_dims):
    weights = []
    biases = []
    activations = []
    
    for i in range(len(layer_array)):
        node_num = layer_array[i][0]
        weights_of_layer = []
        biases_of_layer = []
        if i == 0:
            last_layer_node_number = input_dims
        else:
            last_layer_node_number = layer_array[i-1][0]
        
        for n in range(0,node_num):
            weights_of_node = []
            for l in range(0, last_layer_node_number):
                weights_of_node.append(1) 
            weights_of_layer.append(weights_of_node)
            biases_of_layer.append(0)
            
        weights.append(weights_of_layer)
        biases.append(biases_of_layer)
        activations.append(layer_array[i][1])
    return [weights, biases, activations]

    
def predict_ratio(data, neural_net):
    weights = neural_net[0]
    biases = neural_net[1]
    activations = neural_net[2]
    
    layer_num = len(weights)
    
    for l in range(0, layer_num):
        data = np.dot(weights[l], data)
        for t in range(len(data)):
            data[t] += biases[l][t]
        if activations[l] == 'sigmoid':
            data = sigmoid(data)
        elif activations[l] == 'relu':
            data = relu(data)
        else:
            # If not identified, do it with sigmoid
            data = sigmoid(data)
            print('activation function', activations[l], 'cannot be found. Sigmoid is used')   
    return data


def predict(data, neural_net):
    data = predict_ratio(data, neural_net)
    
    class_num = len(data)
    
    highest_class = None
    highest_class_probability = -1
    
    for i in range(0, class_num):
        if highest_class == None:
            highest_class = i
            highest_class_probability = data[i]
        elif data[i] > highest_class_probability:
            highest_class = i
            highest_class_probability = data[i]
            
    return highest_class, highest_class_probability


def train_network(X, Y, labels, neural_net, epochs=1000):
    for epoch in range(0, epochs):
        for d in range(0, len(X)):
            prediction = predict_ratio(X[d], neural_net)
            
            # Calculate total error per label
            true_prediction = []
            for i in range(0, len(labels)):
                true_prediction.append(0)
            true_prediction[labels.index(Y[d])] = 1
            
            errors = []
            for t in range(len(prediction)):
                errors.append(true_prediction[t] - prediction[t]) 
            adjust_deriv = errors * sigmoid_deriv(prediction)
            
            for k in range(0, len(adjust_deriv)):
                adjustment = np.dot(X[d], adjust_deriv[k])
                neural_net[0][0][k] += adjustment
    return neural_net
