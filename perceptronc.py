import sys
import csv
import numpy as np


# PERCEPTRONC Fit the sequential perceptron algorithm to transformed data
# PERCEPTRONC inputs data with input weights and performs the
# sequential perceptron algorithm in order to find the optimal
# line separating the two classes
# However it first transforms the data using (X - 1.5 - X)**2
# w_init is the initialization of the weight vector
#
# X is the input vector of features
#
# Y is the vector of class labels
#
def perceptronc(w_init, X, Y):
    X = (1.5 - X) ** 2
    k = 0
    e = 0
    misclassified = [1] * len(X)
    w = list(w_init)
    while (sum(misclassified) > 0 and e < 15):
        if k == (len(X)-1):
            e += 1
        x = X[k]
        y = Y[k]
        if (w[0]+w[1]*x)*y > 0:
            misclassified[k] = 0
        else:
            misclassified[k] = 1
            w[0] = w[0] + y
            w[1] = w[1] + x*y
        k = (k+1) % len(X)
    return (w, e)


def main():
    rfile = sys.argv[1]

    # read in csv file into np.arrays X1, X2, Y1, Y2
    csvfile = open(rfile, 'rb')
    dat = csv.reader(csvfile, delimiter=',')
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for i, row in enumerate(dat):
        if i > 0:
            X1.append(float(row[0]))
            X2.append(float(row[1]))
            Y1.append(float(row[2]))
            Y2.append(float(row[3]))

    X1 = np.array(X1)
    X2 = np.array(X2)
    Y1 = np.array(Y1)
    Y2 = np.array(Y2)
    w_init = [0] * 2
    # perceptronc(w_init, X1, Y1)
    perceptronc(w_init, X2, Y2)


if __name__ == "__main__":
    main()
