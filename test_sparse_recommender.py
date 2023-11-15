import pytest as pt
import sparse_recommender as sprec


class TestSparse:
    def test_set(self):
        simple_matrix = sprec.SparseMatrix()
        simple_matrix.set(0, 0, 5)
        simple_matrix.set(1, 1, 2)
        simple_matrix.set(2, 1, 9)
        simple_matrix.set(2, 2, 10)
        simple_matrix.set(3, 0, 2)
        simple_matrix.set(3, 3, 8)

        assert simple_matrix.get(2, 1) == 9
        assert simple_matrix.get(3, 0) == 2

        with pt.raises(ValueError) as result:
            simple_matrix.set(-1, 4 , 20)
        assert str(result.value) == "Row or Column values are invalid pass the valid values"

        with pt.raises(ValueError) as result:
            simple_matrix.set(3, -3, 11)
        assert str(result.value) == "Row or Column values are invalid pass the valid values"




    def test_get(self):
        simple_matrix = sprec.SparseMatrix()
        simple_matrix.set(0, 0, 5)
        simple_matrix.set(1, 1, 2)
        simple_matrix.set(2, 1, 9)
        assert simple_matrix.get(1, 1) == 2
        assert simple_matrix.get(1, 0) == 0
        assert simple_matrix.get(2, 1) == 9
        with pt.raises(ValueError) as result:
            simple_matrix.get(4, 4)
        assert str(result.value) == "Row or Column values are invalid pass the valid values"

        with pt.raises(ValueError) as result:
            simple_matrix.get(1, 4)
        assert str(result.value) == "Row or Column values are invalid pass the valid values"





    def test_recommend(self):
        recommand_matrix = sprec.SparseMatrix()
        recommand_matrix.set(0, 0, 5)
        recommand_matrix.set(0, 1, 2)
        recommand_matrix.set(1, 0, 2)
        recommand_matrix.set(1, 1, 1)
        recommand_matrix.set(1, 2, 2)
        recommand_matrix.set(2, 0, 3)
        recommand_matrix.set(2, 1, 1)
        recommand_matrix.set(2, 2, 1)
        with pt.raises(ValueError) as result:
            recommand_matrix.set( -1, 2, 7)
        assert str(result.value) == 'Row or Column values are invalid pass the valid values'

        vector = [0, 5, 8]
        recommendations = recommand_matrix.recommend(vector)
        assert recommendations == {(0, 0): 0, (0, 1): 10, (1, 0): 0, (1, 1): 5, (1, 2): 16, (2, 0): 0, (2, 1): 5, (2, 2): 8}

        vector1 = [0, 5, 8, 5]
        with pt.raises(ValueError) as result:
            recommand_matrix.recommend(vector1)
        assert str(result.value) == "Length of the vector not matched with column size fo a sparse-matrix"



        vector2 = [0]
        with pt.raises(ValueError) as result:
            recommand_matrix.recommend(vector2)
        assert str(result.value) == "Length of the vector not matched with column size fo a sparse-matrix"

        recommand_matrix1 = sprec.SparseMatrix()
        recommand_matrix1.set(0, 0, 1)
        recommand_matrix1.set(1, 0, 1)
        recommand_matrix1.set(1, 1, 1)
        recommand_matrix1.set(1, 2, 1)
        recommand_matrix1.set(2, 0, 1)
        recommand_matrix1.set(2, 2, 1)
        vector = [1, 1, 2]
        recommendations1 = recommand_matrix1.recommend(vector)
        assert recommendations1 == {(0, 0): 1, (1, 0): 1, (1, 1): 1, (1, 2): 2, (2, 0): 1, (2, 2): 2}




    def test_add_movie(self):
        simple_matrix = sprec.SparseMatrix()
        simple_matrix.set(0, 0, 1)
        simple_matrix.set(1, 0, 2)
        simple_matrix.set(1, 1, 2)
        simple_matrix.set(2, 2, 3)
        new_row = [4, 5, 3]
        result = simple_matrix.add_movie(new_row)
        assert result.simpleMatrix == {(0, 0): 1, (1, 0): 2, (1, 1): 2, (2, 2): 3, (3, 0): 4, (3, 1): 5, (3, 2): 3}

        new_row2 = [4, 5]

        with pt.raises(ValueError) as result:
            simple_matrix.add_movie(new_row2)
        assert str(result.value) == "Sparse sparse-matrix indexes are miss matched"

        new_row3 = [4, 5, -1]
        result = simple_matrix.add_movie(new_row3)
        assert result.simpleMatrix == {(0, 0): 1, (1, 0): 2, (1, 1): 2, (2, 2): 3, (3, 0): 4, (3, 1): 5, (3, 2): -1}



    def test_dense(self):
        simplematrix = sprec.SparseMatrix()
        simplematrix.set(0, 0, 1)
        simplematrix.set(1, 0, 2)
        simplematrix.set(1, 1, 2)
        simplematrix.set(2, 2, 3)
        new_rows = [4, 5, 5]

        result = simplematrix.add_movie(new_rows)
        assert result.to_dense() == [[1, 0, 0], [2, 2, 0], [0, 0, 3], [4, 5, 5]]

        new_row2 = [2, 1, 4, 5]

        with pt.raises(ValueError) as result:
            simplematrix.add_movie(new_row2)
        assert str(result.value) == "Sparse sparse-matrix indexes are miss matched"

        new_rows = [100, 2, -1]

        result = simplematrix.add_movie(new_rows)
        assert result.to_dense() == [[1, 0, 0], [2, 2, 0], [0, 0, 3], [100, 2, -1]]




    def tests_parse_matrix(self):
        matrix = sprec.SparseMatrix()
        matrix.set(0, 0, 1)
        matrix.set(1, 1, 4)
        matrix.set(2, 2, 3)

        assert matrix.get(0, 0) == 1
        assert matrix.get(1, 1) == 4
        assert matrix.get(2, 2) == 3

        vector = [2, 3, 1]
        recommendations = matrix.recommend(vector)
        assert recommendations == {(0, 0): 2, (1, 1): 12, (2, 2): 3}

        new_rows = [2, 1, 1]
        merged_matrix = matrix.add_movie(new_rows)
        assert merged_matrix.get(3, 0) == 2
        assert merged_matrix.get(3, 2) == 1

        dense_matrix = merged_matrix.to_dense()
        assert dense_matrix == [[1, 0, 0], [0, 4, 0], [0, 0, 3], [2, 1, 1]]



    def test_sparse_matrix2(self):
        matrix = sprec.SparseMatrix()
        matrix.set(0, 0, 5)
        matrix.set(1, 0, 3)
        matrix.set(1, 1, 6)
        matrix.set(1, 2, 10)
        matrix.set(2, 2, 3)
        matrix.set(3, 3, 6)

        with pt.raises(ValueError) as result:
            matrix.set(-99, 4 , 20)
        assert str(result.value) == "Row or Column values are invalid pass the valid values"


        assert matrix.get(0, 0) == 5
        assert matrix.get(1, 1) == 6
        assert matrix.get(2, 2) == 3
        vector = [1]
        with pt.raises(ValueError) as result:
            matrix.recommend(vector)
        assert str(result.value) == "Length of the vector not matched with column size fo a sparse-matrix"

        new_rows = [4, 5, 5, 5]
        merged_matrix = matrix.add_movie(new_rows)
        assert merged_matrix.simpleMatrix == {(0, 0): 5, (1, 0): 3, (1, 1): 6, (1, 2): 10, (2, 2): 3, (3, 3): 6, (4, 0): 4, (4, 1): 5, (4, 2): 5, (4, 3): 5}
        assert merged_matrix.get(1, 2) == 10
        assert merged_matrix.get(3, 3) == 6

        dense_matrix = merged_matrix.to_dense()
        assert dense_matrix == [[5, 0, 0, 0], [3, 6, 10, 0], [0, 0, 3, 0], [0, 0, 0, 6], [4, 5, 5, 5]]
