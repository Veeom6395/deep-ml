import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Standardization (population standard deviation, ddof=0)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    standardized_data = np.divide(
        data - mean,
        std,
        out=np.zeros_like(data, dtype=float),
        where=std != 0
    )

    # Min-max normalization
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    ranges = max_vals - min_vals

    normalized_data = np.divide(
        data - min_vals,
        ranges,
        out=np.zeros_like(data, dtype=float),
        where=ranges != 0
    )

    # Round results to 4 decimal places
    standardized_data = np.round(standardized_data, 4)
    normalized_data = np.round(normalized_data, 4)

    return standardized_data, normalized_data