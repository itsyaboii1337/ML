import numpy as np

def compute_loss(y, tx, w):
    
    N = y.shape[0]
    e = y - np.dot(tx, w)
    L = (1/(2*N))*np.dot(e.transpose(), e)
    
    return L

def compute_gradient(y, tx, w):
    
    N = y.size
    
    return - np.matmul(np.transpose(tx), y-np.matmul(tx, w))/N

def compute_st_gradient(y_n, tx_n, w):
    """ We don't use batches here! """
    return -np.dot(tx_n.transpose(), y_n - np.matmul(tx_n, w))

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