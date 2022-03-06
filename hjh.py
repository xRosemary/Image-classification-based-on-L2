import pandas as pd
import numpy as np

import scipy.io as sio
from scipy.linalg import norm
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def readData(test_size):
    domains = ['amazon', 'caltech', 'dslr', 'webcam']
    X_all = np.empty((0, 4096))
    y_all = np.empty((0),dtype=np.int)
    for domain in domains:
        data = sio.loadmat('OC10_vgg6/' + domain + '.mat')  # source domain
        Xs, ys = data['FTS'].astype(np.float64), data['LABELS'].ravel()
        ys = LabelEncoder().fit(ys).transform(ys).astype(np.int)
        Xs = Xs / norm(Xs, axis=1, keepdims=True)
        X_all = np.concatenate((X_all,Xs))
        y_all = np.concatenate((y_all,ys))
    return train_test_split(X_all, y_all, test_size=test_size)

test_size = 0.2
X,X_test,y,y_test = readData(test_size)

learning_rate = 0.1
n_features = X.shape[1]
n_samples = X.shape[0]
n_classes = len(np.unique(y))

std = 1e-3
W = np.random.normal(loc=0.0, scale=std, size=(n_features, n_classes))
b = np.zeros(n_classes)
t = 0
maxt = 1000
last_loss = 9999


while True:
    loss = 0
    for index in range(n_samples):
        scores = np.dot(X, W) + b
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[np.arange(n_samples), y])
        loss = np.sum(correct_logprobs) / n_samples
        dscores = probs.copy()
        dscores[np.arange(n_samples), y] -= 1
        dscores /= n_samples
        dW = X.T.dot(dscores)
        db = np.sum(dscores, axis=0, keepdims=True)
        W = W - learning_rate * dW
        b = b - learning_rate * db
        print(loss)
        if abs(loss - last_loss) < 1e-3:
            break
        last_loss = loss

    t += 1
    if t > maxt:
        break

y_test_socre  = np.dot(X_test,W) + b
y_test_exp_scores = np.exp(y_test_socre)
y_test_probs = y_test_exp_scores / np.sum(y_test_exp_scores, axis=1, keepdims=True)
y_test_probs_max = np.argmax(y_test_probs, axis=1)

print(classification_report(y_test, y_test_probs_max))
