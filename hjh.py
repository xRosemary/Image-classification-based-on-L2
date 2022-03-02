import pandas as pd
import numpy as np

import scipy.io as sio
from scipy.linalg import norm
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def readData(sc, tg):
    data = sio.loadmat('Datasets/OC10_vgg6/' + sc + '.mat')  # source domain
    Xs, ys = data['FTS'].astype(np.float64), data['LABELS'].ravel()
    ys = LabelEncoder().fit(ys).transform(ys).astype(np.int)

    data = sio.loadmat('Datasets/OC10_vgg6/' + tg + '.mat')  # target domain
    Xt, yt = data['FTS'].astype(np.float64), data['LABELS'].ravel()
    yt = LabelEncoder().fit(yt).transform(yt).astype(np.int)

    Xs = Xs / norm(Xs, axis=1, keepdims=True)
    Xt = Xt / norm(Xt, axis=1, keepdims=True)
    return np.array(Xs), np.array(ys), np.array(Xt), np.array(yt)
domains = ['amazon', 'caltech', 'dslr', 'webcam']
Xs, ys, Xt, yt = readData(domains[0], domains[1])

X = Xs
y = ys

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



