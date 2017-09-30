'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score


def predict():

    # Load Data
    filename = 'data/test_potus_by_county.csv'
    data = np.loadtxt(filename, delimiter=',')
    Xtest = data[:, 1:]
    ytest = np.array([data[:, 0]]).T
    n, d = Xtest.shape

    normalmean = []

    # get classifier from build_model.py
    y_pred = clf.predict(Xtest)

    normalmean.append(accuracy_score(ytest, y_pred))

    meanDecisionTreeAccuracy = np.average(np.array(normalmean))
    stddevDecisionTreeAccuracy = np.std(np.array(normalmean))

    #return predictions


if __name__ == "__main__":
    predict()

