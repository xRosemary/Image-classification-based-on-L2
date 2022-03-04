import numpy as np
import scipy.io as sio
from scipy.linalg import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def readData(test_size):
    domains = ['amazon', 'caltech', 'dslr', 'webcam']

    x_sum = np.empty((0, 4096))
    y_sum = np.empty((1, 0))

    for name in domains:
        data = sio.loadmat('Datasets/OC10_vgg6/' + name + '.mat')  # source domain
        Xs, ys = data['FTS'].astype(np.float64), data['LABELS'].ravel()
        ys = LabelEncoder().fit(ys).transform(ys).astype(np.float64)
        Xs = Xs / norm(Xs, axis=1, keepdims=True)
        
        # 矩阵拼接
        x_sum = np.r_[x_sum, np.array(Xs)]
        y_sum = np.append(y_sum, np.array(ys))

    return train_test_split(x_sum, y_sum, test_size=test_size)
