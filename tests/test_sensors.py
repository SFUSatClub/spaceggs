import pytest

import math
import numpy as np
from numpy.testing import assert_allclose
import transforms3d as tf

from spaceggs.models import SunSensor

def test_sunsensor():
    sensor_position = np.array([1,1,1])
    sensor_orientation = np.array([0,0,1,0]) # no rotation
    sun_position = np.array([10,0,0,1])
    expected = 1
    actual = SunSensor(sensor_position, sensor_orientation).observe(sun_position)
    
    assert_allclose(actual, expected, rtol=1e-02)