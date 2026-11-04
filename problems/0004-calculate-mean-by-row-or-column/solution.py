def calculate_matrix_mean(matrix, mode):
    if mode == 'row':
        return [sum(row) / len(row) for row in matrix]

    elif mode == 'column':
        rows = len(matrix)
        cols = len(matrix[0])

        return [
            sum(matrix[i][j] for i in range(rows)) / rows
            for j in range(cols)
        ]

    else:
        raise ValueError("Mode must be 'row' or 'column'")
