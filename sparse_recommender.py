class SparseMatrix:
    def __init__(sparsematrix):
        sparsematrix.simpleMatrix = {}

    def set(sparsematrix, row, col, value):
        if len(sparsematrix.simpleMatrix) !=0:
            numrows = max(row for row, _ in sparsematrix.simpleMatrix.keys()) + 1
            numcols = max(col for _, col in sparsematrix.simpleMatrix.keys()) + 1
            if row < 0 or row > numrows or col < 0 or col > numcols:
                raise ValueError("Row or Column values are invalid pass the valid values")
        if value != 0:
            sparsematrix.simpleMatrix[(row, col)] = value
        elif (row, col) in sparsematrix.simpleMatrix:
            del sparsematrix.simpleMatrix[(row, col)]

    def get(sparsematrix, row, col):
        numrows = max(row for row, _ in sparsematrix.simpleMatrix.keys()) + 1
        numcols = max(col for _, col in sparsematrix.simpleMatrix.keys()) + 1
        if row < 0 or row > numrows or col < 0 or col > numcols:
            raise ValueError("Row or Column values are invalid pass the valid values")
        return sparsematrix.simpleMatrix.get((row, col), 0)

    def recommend(sparsematrix, vector):
        recommendation = {}
        cols = max(col for _, col in sparsematrix.simpleMatrix.keys()) + 1
        if len(vector) != cols:
            raise ValueError("Length of the vector not matched with column size fo a sparse-matrix")
        for (row, col), value in sparsematrix.simpleMatrix.items():
            recommendation[(row, col)] = value * vector[col]
        return recommendation

    def add_movie(sparsematrix, new_row):
        rows = max(row for row, _ in sparsematrix.simpleMatrix.keys()) + 1
        cols = max(col for _, col in sparsematrix.simpleMatrix.keys()) + 1
        if len(new_row) != cols:
            raise ValueError("Sparse sparse-matrix indexes are miss matched")
        resultMatrix = SparseMatrix()
        for (row, col), value in sparsematrix.simpleMatrix.items():
            resultMatrix.set(row, col, value)
        for col, value in enumerate(new_row):
            resultMatrix.set(rows, col, value)
        return resultMatrix

    def to_dense(sparsematrix):
        max_row = max(row for row, _ in sparsematrix.simpleMatrix.keys()) + 1
        max_col = max(col for _, col in sparsematrix.simpleMatrix.keys()) + 1
        dense_matrix = [[0] * max_col for _ in range(max_row)]
        for (row, col), value in sparsematrix.simpleMatrix.items():
            dense_matrix[row][col] = value
        return dense_matrix
