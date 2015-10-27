import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Rohan Agarwal'


# ZIP_SHUF Shuffle separate arrays together
# ZIP_SHUF takes two arrays and performs the following operations:
# 1) Zip them together
# 2) Shuffle their order
# 3) Unzip to get the shuffled arrays with corresponding indices
def zip_shuf(X_poly, Y):
    shuf = zip(X_poly, Y)
    np.random.shuffle(shuf)
    X_poly = np.array([x[0] for x in shuf])
    Y_shuf = np.array([x[1] for x in shuf])
    return X_poly, Y_shuf


# TRAIN_TEST Create training and testing data
# TRAIN_TEST takes the data and creates training and testing data
# The train and test split also depends on the number of polynomials
# As well as the number of folds of cross validation
# And the current fold being split on
def train_test(X_poly, Y, i, j, n):
    X_mat = X_poly[:, 0:i + 1]
    low_idx = j * len(X_poly) / n
    high_idx = (j + 1) * len(X_poly) / n
    X_test = X_mat[low_idx:high_idx, :]
    X_train = np.concatenate(
        (X_mat[0:low_idx, :], X_mat[high_idx:, :]), axis=0)
    Y_test = Y[low_idx:high_idx]
    Y_train = np.concatenate((Y[0:low_idx], Y[high_idx:, ]))
    return X_train, X_test, Y_train, Y_test

# REGRESS Fit a least squares regression model
# REGRESS implements the closed form solution to linear regression
def regress(X_train, Y_train):
    X_train_t = np.transpose(X_train)
    X_train_inv = np.linalg.inv(np.dot(X_train_t, X_train))
    X_term = np.dot(X_train_inv, X_train_t)
    w = np.dot(X_term, Y_train)
    return w

#   NFOLDPOLYFIT Fit polynomial of the best degree to data.
#   NFOLDPOLYFIT(X,Y,maxDegree, nFold, verbose) finds and returns the coefficients
#   of a polynomial P(X) of a degree between 1 and N that fits the data Y
#   best in a least-squares sense, averaged over nFold trials of cross validation.
#
#   P is a vector (in numpy) of length N+1 containing the polynomial coefficients in
#   descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). use
#   numpy.polyval(P,Z) for some vector of input Z to see the output.
#
#   X and Y are vectors of datapoints specifying  input (X) and output (Y)
#   of the function to be learned. Class support for inputs X,Y:
#   float, double, single
#
#   maxDegree is the highest degree polynomial to be tried. For example, if
#   maxDegree = 3, then polynomials of degree 0, 1, 2, 3 would be tried.
#
#   nFold sets the number of folds in nfold cross validation when finding
#   the best polynomial. Data is split into n parts and the polynomial is run n
#   times for each degree: testing on 1/n data points and training on the
#   rest.
#
#   verbose, if set to 1 shows mean squared error as a function of the
#   degrees of the polynomial on one plot, and displays the fit of the best
#   polynomial to the data in a second plot.
def nfoldpolyfit(X, Y, maxK, n, verbose):
    X_poly = []
    for i in range(maxK + 1):
        X_poly.append(X ** i)

    X_poly = np.transpose(np.array(X_poly))
    MSE_values = []
    all_weights = []

    X_poly, Y_shuf = zip_shuf(X_poly, Y)

    for i in range(maxK + 1):
        MSE_this_poly = []
        weights_this_poly = []
        for j in range(n):
            X_train, X_test, Y_train, Y_test = train_test(
                X_poly, Y_shuf, i, j, n)
            w = regress(X_train, Y_train)
            weights_this_poly.append(w)
            y_i = np.dot(X_test, w)
            MSE = sum((Y_test - y_i) ** 2) / len(y_i)
            MSE_this_poly.append(MSE)
        MSE_values.append(MSE_this_poly)
        all_weights.append(weights_this_poly)

    MSE_all_poly = [sum(x) / len(x) for x in MSE_values]
    best_poly = MSE_all_poly.index(min(MSE_all_poly))
    best_line_location = MSE_values[best_poly].index(
        min(MSE_values[best_poly]))
    best_line = all_weights[best_poly][best_line_location]

    print "The optimal weights are: " + str(best_line)

    if verbose:
        plt.scatter(range(0, maxK + 1), [sum(x) / len(x) for x in MSE_values])
        plt.xlabel('Polynomial degree')
        plt.ylabel('MSE')
        plt.title('MSE vs Degree of Polynomial Regression')
        plt.show()

        X_new = np.linspace(min(X), max(X), 1000)
        Y_new = [0] * len(X_new)
        for i, val in enumerate(best_line):
            Y_new += val * (X_new ** i)

        plt.scatter(X, Y)
        plt.plot(X_new, Y_new, '--r')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Scatterplot and Regression Line for X & Y")
        plt.show()

    return MSE_values


def main():
    rfile = sys.argv[1]
    maxK = int(sys.argv[2])
    nFolds = int(sys.argv[3])
    verbose = bool(sys.argv[4])

    csvfile = open(rfile, 'rb')
    dat = csv.reader(csvfile, delimiter=',')
    X = []
    Y = []
    # put the x coordinates in the list X, the y coordinates in the list Y
    for i, row in enumerate(dat):
        if i > 0:
            X.append(float(row[0]))
            Y.append(float(row[1]))

    X = np.array(X)
    Y = np.array(Y)
    nfoldpolyfit(X, Y, maxK, nFolds, verbose)

if __name__ == "__main__":
    main()
