import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros(len(b), dtype=float)

    for _ in range(n):
        x_new = np.zeros_like(x)

        for i in range(len(b)):
            sum_other = sum(A[i][j] * x[j] for j in range(len(b)) if j != i)
            x_new[i] = (b[i] - sum_other) / A[i][i]

        x = x_new

    return np.round(x, 4).tolist()