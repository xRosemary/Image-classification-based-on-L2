from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.optimize import minimize
from scipy._lib._util import _asarray_validated
from com.MainProject1 import ShowData

domains = ['amazon', 'caltech', 'dslr', 'webcam']

X_train, y_train, X_test, y_test = ShowData.readData(domains[2], domains[3])


# def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
#     a = _asarray_validated(a, check_finite=False)
#     if b is not None:
#         a, b = np.broadcast_arrays(a, b)
#         if np.any(b == 0):
#             a = a + 0.  # promote to at least float
#             a[b == 0] = -np.inf
#
#     a_max = np.amax(a, axis=axis, keepdims=True)
#
#     if a_max.ndim > 0:
#         a_max[~np.isfinite(a_max)] = 0
#     elif not np.isfinite(a_max):
#         a_max = 0
#
#     if b is not None:
#         b = np.asarray(b)
#         tmp = b * np.exp(a - a_max)
#     else:
#         tmp = np.exp(a - a_max)
#
#     # suppress warnings about log of zero
#     with np.errstate(divide='ignore'):
#         s = np.sum(tmp, axis=axis, keepdims=keepdims)
#         if return_sign:
#             sgn = np.sign(s)
#             s *= sgn  # /= makes more sense but we need zero -> zero
#         out = np.log(s)
#
#     if not keepdims:
#         a_max = np.squeeze(a_max, axis=axis)
#     out += a_max
#
#     if return_sign:
#         return out, sgn
#     else:
#         return out


def softmax(Z):
    return np.exp(-Z) / np.sum(np.exp(-Z))


def cost(input_x, truth, thetas):
    hypothesis = softmax(np.dot(input_x, thetas.T))
    difference = hypothesis - truth
    return np.dot(difference, difference.T)


def gradient(X, truth, thetas, learning_rate):
    hypothesis = softmax(np.dot(X, thetas.T))
    d_cost = 2 * (hypothesis - truth)

    grad = 0.0

    for j in range(10):
        if truth[j] == 1:
            grad += d_cost[j] * hypothesis[j]
        else:
            grad -= d_cost[j] * pow(hypothesis[j], 2)

    return grad * learning_rate


def creat_learning_data(X, y, label_number, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    # k * (n + 1) array for the parameters of each of the k classifiers
    thetas = np.zeros((label_number, params + 1))
    # print(thetas.shape)  # shape = 10,4097

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # (X dot theta.T) shape = (k, 10)

    for index in range(rows):
        truth = np.array([1 if label == y[index] else 0 for label in range(1, label_number + 1)])
        input_x = X[index]
        grad = gradient(input_x, truth, thetas, learning_rate)

        thetas -= grad
        print(cost(input_x, truth, thetas))
        # if cost(input_x, truth, thetas)<=0.05:
        #     print(True)
        #     break
        # print(cost(input_x, truth, thetas))
        # fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        # all_theta[i - 1, :] = fmin.x

    np.savetxt('all_theta', thetas)
    return thetas


def predict(input_x, thetas):
    hypothesis = softmax(np.dot(input_x, thetas.T))
    print(hypothesis)


label_number = len(set(y_train))
thetas = creat_learning_data(X_train, y_train, label_number, learning_rate=0.1)
# rows = X_test.shape[0]
# X_test = np.insert(X_test, 0, values=np.ones(rows), axis=1)
# predict(X_test[0],thetas)