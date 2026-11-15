
import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.

    Args:
        A: A 2x2 numpy array

    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
    """
    # Step 1: Compute A transpose times A
    B = A.T @ A

    # Step 2: Compute Jacobi rotation angle
    if abs(B[0, 1]) < 1e-15:
        c, s = 1.0, 0.0
    else:
        tau = (B[1, 1] - B[0, 0]) / (2.0 * B[0, 1])

        t = np.sign(tau) / (abs(tau) + np.sqrt(1 + tau**2))
        if tau == 0:
            t = 1.0

        c = 1.0 / np.sqrt(1 + t**2)
        s = t * c

    # Step 3: Construct Jacobi rotation
    V = np.array([
        [c, s],
        [-s, c]
    ])

    # Step 4: Diagonalize A.T @ A
    D = V.T @ B @ V

    # Step 5: Compute singular values
    S = np.sqrt(np.maximum(np.diag(D), 0))

    # Step 6: Sort singular values in descending order
    idx = np.argsort(S)[::-1]
    S = S[idx]
    V = V[:, idx]

    # Step 7: Compute left singular vectors
    U = A @ V / S

    # Step 8: Return decomposition
    return U, S, V.T
