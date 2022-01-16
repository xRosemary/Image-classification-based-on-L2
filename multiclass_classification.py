import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import classification_report
from com.MainProject1 import ShowData

domains = ['amazon', 'caltech', 'dslr', 'webcam']

X_train, y_train, X_test, y_test = ShowData.readData(domains[2], domains[3])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()


def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    # k * (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x

    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax


def creat_learning_data():
    rows = X_train.shape[0]
    params = X_train.shape[1]

    all_theta = np.zeros((10, params + 1))

    X = np.insert(X_train, 0, values=np.ones(rows), axis=1)

    theta = np.zeros(params + 1)

    y_0 = np.array([1 if label == 0 else 0 for label in y_train])
    y_0 = np.reshape(y_0, (rows, 1))

    # print(np.unique(y_train))
    all_theta = one_vs_all(X_train, y_train, 10, 0.1)
    # # print(all_theta)
    np.savetxt('all_theta', all_theta)


def load_learning_data():
    return np.loadtxt('all_theta')


# creat_learning_data()

all_theta = load_learning_data()

y_pred = predict_all(X_test, all_theta)
# print(y_pred)
print(classification_report(y_test, y_pred))