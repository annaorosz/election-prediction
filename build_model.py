import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

def dt(numTrials=100):

    # Load Data
    filename = 'data/train_potus_by_county.csv'
    data = np.loadtxt(filename, delimiter=',')
    Xtrain = data[:, 1:]
    ytrain = np.array([data[:, 0]]).T
    n, d = Xtrain.shape

    # split data into 10 "equal" parts
    rem = n % 10
    floor = n / 10
    folds = []
    for k in range(rem):
        folds.append(floor + 1)
    for l in range(10 - rem):
        folds.append(floor)

    start = []
    stop = []

    index = 0
    for a in folds:
        if len(start) <= 0:
            start.append(1)
        else:
            start.append(stop[index - 1] + 1)
        if len(stop) <= 0:
            stop.append(a)
        else:
            stop.append(stop[index - 1] + a)
        index += 1

    for i in range(100):

        # shuffle the data
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        Xtrain = X[:, :]
        ytrain = y[:, :]
        # train the decision tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(Xtrain, ytrain)

        return clf

if __name__ == "__main__":
    dt()