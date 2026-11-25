import numpy as np

def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> (np.ndarray, float, list[float]):

    weights = initial_weights.astype(float).copy()
    bias = float(initial_bias)
    mse_values = []

    n = len(features)

    for _ in range(epochs):
        # Forward pass
        z = np.dot(features, weights) + bias
        predictions = 1.0 / (1.0 + np.exp(-z))

        # Compute MSE before updating parameters
        errors = predictions - labels
        mse = np.mean(errors ** 2)
        mse_values.append(round(float(mse), 4))

        # Backpropagation
        # Derivative of sigmoid
        sigmoid_derivative = predictions * (1.0 - predictions)

        # Gradients of MSE loss
        delta = 2.0 * errors * sigmoid_derivative

        dw = np.dot(features.T, delta) / n
        db = np.mean(delta)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

    # Round final parameters
    updated_weights = np.round(weights, 4)
    updated_bias = round(float(bias), 4)

    return updated_weights, updated_bias, mse_values