import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    probabilities = []

    for feature_vector in features:
        # Calculate weighted sum
        z = sum(x * w for x, w in zip(feature_vector, weights)) + bias

        # Apply sigmoid activation
        probability = 1 / (1 + math.exp(-z))
        probabilities.append(round(probability, 4))

    # Calculate Mean Squared Error
    mse = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)

    return probabilities, round(mse, 4)