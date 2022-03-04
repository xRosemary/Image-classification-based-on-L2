import numpy as np
from com.MainProject1 import LoadData
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = LoadData.readData(test_size=0.3)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


def cost(X, y, thetas, label_number):
    cost_value = 0

    for index in range(len(X)):
        truth = np.array([1 if label == y[index] else 0 for label in range(label_number)])
        hypothesis = softmax(np.dot(X[index], thetas.T))
        difference = hypothesis - truth
        cost_value += np.dot(difference, difference.T)
    return cost_value / len(X)


def gradient(X, truth, thetas):
    hypothesis = softmax(np.dot(X, thetas.T))
    grad = (2 * (hypothesis - truth) * (hypothesis - np.power(hypothesis, 2))).reshape(-1, 1).dot(X.reshape(1, -1))
    return grad


def creat_learning_data(X, y, learning_rate, max_step, max_cost):
    label_number = len(set(y))

    rows = X.shape[0]
    params = X.shape[1]
    thetas = np.zeros((label_number, params + 1))
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    for step in range(max_step):
        for index in range(rows):
            truth = np.array([1 if label == y[index] else 0 for label in range(label_number)])
            d_theta = gradient(X[index], truth, thetas) * learning_rate
            thetas -= d_theta

        cost_value = cost(X, y, thetas, label_number)
        print("after: ", cost_value)

        if cost_value <= max_cost:
            break

    np.savetxt('all_theta', thetas)
    return thetas


def predict(input_x, thetas):
    input_x = np.insert(input_x, 1, 0)
    hypothesis = softmax(np.dot(input_x, thetas.T))
    return np.argmax(hypothesis)


def load_learning_data():
    return np.loadtxt('all_theta')


def predict_all(X, thetas):
    rows = X.shape[0]
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    h = softmax(np.dot(X, thetas.T))
    h_argmax = np.argmax(h, axis=1)

    return h_argmax


if __name__ == "__main__":
    thetas = creat_learning_data(X_train, y_train, learning_rate=0.1, max_step=1000, max_cost=0.001)
    # thetas = load_learning_data()
    y_pred = predict_all(X_test, thetas)
    print(classification_report(y_test, y_pred))
