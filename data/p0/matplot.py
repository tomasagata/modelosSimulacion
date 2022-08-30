import numpy as np
from matplotlib import pyplot as plt
"""
Completar las funciones señaladas con la logica
correspondiente segun conceptos vistos en clase 0. 
No remover ni modifcar la constante X
"""
X = np.arange(10)


def y_values(x, mode):
    """
    Given an array of X values, return the y
    values according to the mode parameter.
    mode can be l for a linear function y = m*x + b
    where b = 2 and m = 3 and q for a quadratic function
    y = x^2
    >>> y_values(X, 'l')
    array([ 2,  5,  8, 11, 14, 17, 20, 23, 26, 29])
    >>> y_values(X, 'q')
    array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])
    """

    if mode == 'l':
        return np.add(
            np.multiply(
                x,
                np.repeat(
                    [3],
                    len(x)
                )
            ),
            np.repeat(
                [2],
                len(x)
            )
        )
    elif mode == 'q':
        return np.multiply(x, x)

    return x


def plot_x_y(x, y, mode):
    """
    Given x and y values of the same size, use matplot 'plt'
    to plot a specific chart. This function can be solved in one line
    """

    plt.plot(x, y, mode)


if __name__ == '__main__':
    # a sample result, try as many as you want
    Y = y_values(X, 'l')
    plot_x_y(X, Y, 'r')
    plt.show()
