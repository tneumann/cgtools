import numpy as np
import numpy.testing as npt
from cgtools.indexing import valid_indices


def test_valid_indices_randomized_tests():
    for axes in range(1, 10):
        shape = np.random.randint(1, 10, axes)
        a = np.empty(shape)
        indices = np.random.randint(-5, 15, (500, axes))
        indices_valid = valid_indices(indices, shape)
        raised = False
        print "--"
        print indices.shape
        print indices_valid.shape
        print indices_valid
        try:
            a[zip(*indices_valid)]
        except IndexError:
            raised = True
        assert not raised

def test_valid_indices_special_input():
    # should also work on 1d input
    npt.assert_array_equal(valid_indices([-1, 1, 2, 3], (2,)), [1])
    npt.assert_array_equal(valid_indices([-1, 1, 2, 3], 2), [1])
    # should work on floats
    npt.assert_array_equal(valid_indices([-1., 1.3, 0.6, 2.2, 3.5], (2,)), [1, 1])
    # should work on empty arrays
    npt.assert_array_equal(valid_indices([], (2, 3)), [])

