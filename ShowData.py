import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.linalg import norm
from sklearn.preprocessing import scale, LabelEncoder


def readData(sc, tg):
    data = sio.loadmat('Datasets/OC10_vgg6/' + sc + '.mat')  # source domain
    Xs, ys = data['FTS'].astype(np.float64), data['LABELS'].ravel()
    ys = LabelEncoder().fit(ys).transform(ys).astype(np.float64)

    data = sio.loadmat('Datasets/OC10_vgg6/' + tg + '.mat')  # target domain
    Xt, yt = data['FTS'].astype(np.float64), data['LABELS'].ravel()
    yt = LabelEncoder().fit(yt).transform(yt).astype(np.float64)

    Xs = Xs / norm(Xs, axis=1, keepdims=True)
    Xt = Xt / norm(Xt, axis=1, keepdims=True)
    return np.array(Xs), np.array(ys+1), np.array(Xt), np.array(yt+1)


# domains = ['amazon', 'caltech', 'dslr', 'webcam']
#
# train_x, train_y, test_x, test_y = readData(domains[0], domains[1])
# train_y = train_y+1
# print(train_y)