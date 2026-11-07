import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    theta = np.linalg.pinv(X.T @ X) @ X.T @ y

    return [round(float(coef), 4) for coef in theta]