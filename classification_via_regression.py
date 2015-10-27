# Simple script to plot the use of regression as a classfier
import numpy as np
import matplotlib.pyplot as plt


def main(n):
    # Create synthetic data
    X = [0, 1, 2, 3, 4, 5, 6, 7, 8, n]
    Y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    best_line = list(reversed(np.polyfit(X, Y, 1)))

    new_X = np.linspace(min(X), max(X), 100)
    new_Y = [0] * len(new_X)
    for i, val in enumerate(best_line):
            new_Y += val*(new_X ** i)
    vertical_X = [float((0.5-best_line[0])/best_line[1])] * len(new_X)
    horizontal_Y = [float(max(Y)-min(Y))/2] * len(new_X)

    plt.scatter(X, Y)
    plt.plot(new_X, new_Y, 'r')
    plt.plot(vertical_X, new_Y, '--k')
    plt.plot(new_X, horizontal_Y, '--k')
    plt.plot([])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Classification via Linear Regression")
    plt.show()

main(9)
main(30)
