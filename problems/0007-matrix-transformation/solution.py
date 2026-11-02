import numpy as np

def transform_matrix(A, T, S):
    try:
        A = np.array(A, dtype=float)
        T = np.array(T, dtype=float)
        S = np.array(S, dtype=float)

        # Check matrix dimensions
        if (T.shape[0] != T.shape[1] or
            S.shape[0] != S.shape[1] or
            A.shape != T.shape or
            A.shape != S.shape):
            return -1

        # Check if T and S are invertible
        if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
            return -1

        # Perform transformation T^(-1) * A * S
        result = np.linalg.inv(T) @ A @ S

        return result.tolist()

    except (ValueError, np.linalg.LinAlgError):
        return -1
