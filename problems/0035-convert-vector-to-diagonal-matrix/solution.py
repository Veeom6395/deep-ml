
import numpy as np

def make_diagonal(x):
    n = len(x)
    output = np.zeros((n, n))

    for i in range(n):
        output[i][i] = x[i]

    return output
