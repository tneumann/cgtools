import numpy as np
import numpy.testing as npt
from cgtools.circular import wrapped_distance

def test_wrapped_distance():
    npt.assert_allclose(
        wrapped_distance([0.1, 0.9, 0.2, 0.0, 0.5, 0.2, 0.25, 0.3], 
                         [0.9, 0.1, 0.3, 0.8, 0.5, 0.8, 0.75, 0.7], max_value=1), 
                         [0.2, 0.2, 0.1, 0.2, 0.0, 0.4, 0.5 , 0.4])
