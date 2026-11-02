def reshape_matrix(a, new_shape):
    rows, cols = new_shape

    # Flatten the original matrix
    elements = [x for row in a for x in row]

    # Check whether reshaping is possible
    if len(elements) != rows * cols:
        return []

    # Construct the reshaped matrix
    result = []
    for i in range(rows):
        result.append(elements[i * cols : (i + 1) * cols])

    return result


# Example
a = [[1, 2, 3, 4], [5, 6, 7, 8]]
new_shape = (4, 2)

print(reshape_matrix(a, new_shape))