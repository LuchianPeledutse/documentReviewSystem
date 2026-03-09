import sys
import random
sys.path.append("C:\\main\\GitHub\\documentReviewSystem")

from data_utils import normalize_vectors

import pytest
import numpy as np

random_seed = 42
rng = np.random.default_rng(random_seed)

# Vector normalization test
NUM_TESTS = 5_000
@pytest.mark.parametrize("vector_matrix", [rng.random((10, random.randint(1, 101))) for _ in range(NUM_TESTS)])
def test_vector_normalization(vector_matrix, epsilon = 1e-10) -> np.array:
    """Test whether vectors ara properly normalized with normalize function"""
    normalized_matrix = normalize_vectors(vector_matrix)
    N, E = vector_matrix.shape
    zero_matrix = np.zeros((N, E), dtype = vector_matrix.dtype)
    for row in range(N):
        single_vector = vector_matrix[row, :]
        norm = np.sqrt(sum([value**2 for value in single_vector]))
        norm_single_vector = single_vector/norm
        zero_matrix[row, :] = norm_single_vector
    assert bool((abs(normalized_matrix - zero_matrix) < epsilon).all())
'--------------------------------------------------------------------------------'