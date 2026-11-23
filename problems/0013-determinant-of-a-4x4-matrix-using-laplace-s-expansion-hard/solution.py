def determinant_4x4(matrix: list[list[int | float]]) -> float:
    # Base case: determinant of a 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]

    # Base case: determinant of a 2x2 matrix
    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1]
                - matrix[0][1] * matrix[1][0])

    det = 0

    # Laplace expansion along the first row
    for col in range(len(matrix)):
        # Create the minor by removing the first row and current column
        minor = [
            row[:col] + row[col + 1:]
            for row in matrix[1:]
        ]

        # Apply alternating signs and recursively calculate determinant
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant_4x4(minor)

    return det