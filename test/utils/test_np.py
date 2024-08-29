import numpy as np
import pytest

from soundutils.utils import softmax


@pytest.fixture
def vector():
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def matrix():
    return np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])


@pytest.fixture
def high_precision_vector():
    return np.array([1234567890.0, 1234567891.0, 1234567892.0])


@pytest.mark.unittest
class TestUtilsNp:
    def test_softmax_vector(self, vector):
        result = softmax(vector)
        expected = np.array([0.09003057, 0.24472847, 0.66524096])
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_softmax_matrix(self, matrix):
        result = softmax(matrix, axis=1)
        expected = np.array([
            [0.09003057, 0.24472847, 0.66524096],
            [0.09003057, 0.24472847, 0.66524096]
        ])
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_softmax_matrix_along_default_axis(self, matrix):
        result = softmax(matrix)
        expected = np.array([
            [0.09003057, 0.24472847, 0.66524096],
            [0.09003057, 0.24472847, 0.66524096]
        ])
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_softmax_high_precision(self, high_precision_vector):
        result = softmax(high_precision_vector)
        expected = np.array([0.09003057, 0.24472847, 0.66524096])
        np.testing.assert_almost_equal(result, expected, decimal=8)
