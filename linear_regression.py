from sklearn import datasets
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from random import randint
import mpl_toolkits.mplot3d as plt3d

def linear_regression(x: ndarray, y: ndarray, theta: ndarray) -> None:
    """Given a test set X with labels Y, return the linear regression for the data set
    with the minimum cost."""
    print(cost_function(x, y, theta))
    gradient_descent(x, y, theta, 0.01, 10000)
    print(cost_function(x, y, theta))

def cost_function(x: ndarray, y: ndarray, theta: ndarray) -> float:
    return np.sum(np.square((np.matmul(x, theta)-y)))/(2*x.shape[0])

def gradient_descent(x: ndarray, y: ndarray, theta: ndarray, alpha: float, turns: int) -> None:
    while turns > 0:
        temp = np.zeros([x.shape[1], 1])
        for i in range(x.shape[1]):
            temp[i] = theta[i] - alpha*np.sum(np.matmul(x, theta)-y)/(x.shape[0])
            theta[i] = temp[i]

        turns -= 1

def plotData(x: ndarray, y: ndarray, theta: ndarray):
    plt.plot(x, y, 'bo', x, eval('x*%f+%f' % (theta[1], theta[0])))
    plt.show()

if __name__ == "__main__":
    boston = datasets.load_boston()

    # 2-D data
    X = boston.data[:, 5:6]
    Y = boston.target

    plt.plot(X, Y, 'bo')
    plt.xlabel('Number of Rooms')
    plt.ylabel('Price ($1000s)')
    plt.show()

    theta = np.array([1, X[randint(0, X.shape[0])]])
    X = np.append(np.ones(X.shape), X, axis=1)

    linear_regression(X, Y, theta)

    plt.plot(X, Y, 'bo')
    plt.xlabel('Number of Rooms')
    plt.ylabel('Price ($1000s)')
    plt.plot(X[:, 1], eval('X*%d+%d' % (theta[1], theta[0])))
    plt.show()

    #3-D data
    X = boston.data[:, 4:6]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y)
    ax.set_xlabel('NO2 concentration (parts per 10 million)')
    ax.set_ylabel('Number of rooms')
    ax.set_zlabel('Price ($1000s)')

    plt.show()

    rand = randint(0, X.shape[0])
    theta = np.array([1, X[rand, 0], X[rand, 1]])
    X = np.append(np.ones(X.shape[0]).reshape([X.shape[0], 1]), X, axis=1)

    linear_regression(X, Y, theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 1], X[:, 2], Y)
    ax.set_xlabel('NO2 concentration (parts per 10 million)')
    ax.set_ylabel('Number of rooms')
    ax.set_zlabel('Price ($1000s)')

    theta_points = np.array((eval('%f + %f*X[:, 1] + %f*X[:, 2]' % (theta[0], theta[1], theta[2]))))
    #ax.scatter(X[:, 1], X[:, 2], theta_points)

    #r1 = randint(0, X.shape[0])
    #r2 = randint(0, X.shape[0])
    #r3 = randint(0, X.shape[0])

    #p1 = np.array([X[r1][1], X[r1][2], theta_points[r1]])
    #p2 = np.array([X[r2][1], X[r2][2], theta_points[r2]])
    #p3 = np.array([X[r3][1], X[r3][2], theta_points[r3]])

    #v1 = p2-p1
    #v2 = p3-p1

    #normal = np.cross(v1, v2)
    #d = -np.dot(normal, p1)
    #Z = (-normal[0]*X[:, 1]-normal[1]*X[:, 2]-d)/normal[2]

    ax.plot_trisurf(X[:, 1], X[:, 2], theta_points, color='orange')
    plt.show()













