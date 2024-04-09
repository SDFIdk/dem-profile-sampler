import numpy as np
import pytest

from profilesample.sampler import get_2d_normals

test_data = [
    # single vector
    (np.array([[3.0, 4.0]]), np.array([[-4.0, 3.0]])),
    # multiple vectors
    (np.array([
            [3.0, 4.0],
            [42.0, -1337.0],
            [1.0, 0.0],
        ]),
        np.array([
            [-4.0, 3.0],
            [1337.0, 42.0],
            [0.0, 1.0],
        ])),
]

@pytest.mark.parametrize("vectors,expected_raw", test_data)
def test_normal(vectors, expected_raw):
    expected_normalized = expected_raw / np.hypot(expected_raw[:, 0], expected_raw[:, 1])[:, np.newaxis]
    actual_normals = get_2d_normals(vectors)
    np.testing.assert_array_equal(actual_normals, expected_normalized)
