def matrixmul(a: list[list[int | float]],
              b: list[list[int | float]]) -> list[list[int | float]]:
    if not a or not b or len(a[0]) != len(b):
        return -1

    rows = len(a)
    cols = len(b[0])
    common = len(b)

    c = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            for k in range(common):
                c[i][j] += a[i][k] * b[k][j]

    return c