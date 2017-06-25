import pytest

import math
import numpy as np
from numpy.testing import assert_allclose
import transforms3d as tf

from spaceggs.models import Entity

def test_transformation_matrix():
    position = np.array([2,3,5])
    orientation = np.array([0,0,1,0]) # no rotation
    expected = np.array([
        [1,0,0,2],
        [0,1,0,3],
        [0,0,1,5],
        [0,0,0,1]
    ])    
    
    actual = Entity(position, orientation).M
    
    assert_allclose(actual, expected, rtol=1e-07)