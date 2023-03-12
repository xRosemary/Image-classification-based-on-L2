import pandas as pd
import numpy as np

import scipy.io as sio
from scipy.linalg import norm
from sklearn.preprocessing import scale, LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

encoder = OneHotEncoder(sparse=False)


def readData(domian):
    data = sio.loadmat('OC10_vgg6/' + domian + '.mat')
    Xs, ys = data['FTS'].astype(np.float64), data['LABELS'].ravel()
    ys = LabelEncoder().fit(ys).transform(ys).astype(np.int)
    Xs = Xs / norm(Xs, axis=1, keepdims=True)

    return Xs, ys

def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def loss_func(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def gradient(X, y_true, y_pred):
    grad_W = np.dot(X.T, (y_pred - y_true))
    grad_b =  np.sum(y_pred - y_true, axis=0)
    return grad_W, grad_b

def gradient_descent(X,y,alpha ,iteration):
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    W = np.random.randn(X.shape[1], y_encoded.shape[1])
    b = np.random.randn(y_encoded.shape[1])
    
    
    for i in range(iteration):
        z = np.dot(X, W) + b
        y_pred = softmax(z)

        loss = loss_func(y_encoded, y_pred)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")

        grad_W, grad_b = gradient(X, y_encoded, y_pred)

        W = W - alpha * grad_W
        b = b - alpha * grad_b
    return W,b

domains = ['amazon', 'caltech', 'dslr', 'webcam']
for domain in domains:
    X,y = readData(domain)
    W_optimal ,b_optimal = gradient_descent(X,y,0.01,2000)
#     np.savetxt('W_'+ domain, W_optimal)
#     np.savetxt('b_'+ domain, b_optimal)

    dlist = domains.copy()
    for d in dlist:
        X_test,y_test = readData(d)
        test_z = np.dot(X_test, W_optimal) + b_optimal
        y_test_pred = softmax(test_z)
        y_test_max = np.argmax(y_test_pred, axis=1)
        print("train in "+ domain + "test in " +d)
        print(classification_report(y_test, y_test_max))
