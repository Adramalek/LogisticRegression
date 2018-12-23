import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class LogisticRegression(object):
    def __init__(self, learning_rate=.01, coverage_change=.001, max_steps=10**3):
        self.__learning_rate = learning_rate
        self.__coverage_change = coverage_change
        self.__max_steps = max_steps
        self.w = None

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-np.dot(x, self.w.T)))

    def __J(self, X, y):
        return np.mean(-y * np.log(self.__logistic(X)) - (1.0 - y) * np.log(1.0 - self.__logistic(X)))

    def train(self, X, y):
        self.w = np.zeros(X.shape[1])
        cost = self.__J(X, y)
        delta_cost = 1
        step = 0
        while delta_cost > self.__coverage_change and step < self.__max_steps:
            old_cost = cost
            gradient = np.dot(self.__logistic(X)-y, X)
            self.w -= self.__learning_rate*gradient
            cost = self.__J(X, y)
            delta_cost = old_cost-cost
            step += 1

    def predict(self, X):
        return np.where(self.__logistic(X) >= .5, 1, 0)


def plot(w, X, y):
    x_0 = X[np.where(y == 0)]
    x_1 = X[np.where(y == 1)]

    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0')
    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1')

    reg_line_x = np.arange(0, 1, 0.1)
    reg_line_y = -(w[0]+w[1]*reg_line_x)/w[2]
    plt.plot(reg_line_x, reg_line_y, c='k', label='reg line')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    raw_df = pd.read_csv("dataset1.csv", names=['X1', 'X2', 'Y'])
    normalized_df = (raw_df - raw_df.min()) / (raw_df.max() - raw_df.min())
    train, test = train_test_split(normalized_df, test_size=0.2)

    train_y = train['Y'].values
    train_X = train.drop(['Y'], axis=1)
    train_X.insert(0, 'X0', np.ones(train['X1'].values.shape[0]))
    train_X = train_X.values
    test_y = test['Y'].values
    test_X = test.drop(['Y'], axis=1)
    test_X.insert(0, 'X0', np.ones(test['X1'].values.shape[0]))
    test_X = test_X.values

    lreg = LogisticRegression()
    lreg.train(train_X, train_y)

    pred_y = lreg.predict(test_X)
    plot(lreg.w, train_X, train_y)
    plt.clf()
    plot(lreg.w, test_X, pred_y)
    print('Accuracy=', np.sum(pred_y == test_y)/len(test_y))

    pass

