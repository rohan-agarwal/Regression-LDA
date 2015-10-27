import sys
import csv
import numpy as np


# PERCEPTRONA Fit the sequential perceptron algorithm to data
# PERCEPTRONA inputs data with input weights and performs the
# sequential perceptron algorithm in order to find the optimal
# line separating the two classes
# w_init is the initialization of the weight vector
#
# X is the input vector of features
#
# Y is the vector of class labels
#
def perceptrona(w_init, X, Y):

    k = 0
    e = 0
    misclassified = [1] * len(X)
    w = list(w_init)

    while (sum(misclassified) > 0 and e < 100):

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
    # rfile = 'C:\Users\Rohan\Documents\GitHub\Regression-LDA\linearclass.csv'
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
    perceptrona(w_init, X1, Y1)
    perceptrona(w_init, X2, Y2)

    # Code to plot graphs, commented out
    # plt.scatter(X2, Y2)
    # plt.show()
    # plt.scatter((1.5-X2)**2, Y2)
    # plt.show()


if __name__ == "__main__":
    main()
