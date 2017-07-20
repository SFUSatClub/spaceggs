import pytest

import math
import numpy as np
from numpy.testing import assert_allclose
import transforms3d as tf

from spaceggs.models import SunSensor

def test_sun_front():
    sensor_position = np.array([1,1,1])
    sensor_orientation = np.array([0,0,1,0]) # no rotation
    sun_position = np.array([10,0,0,1])
    expected = 1
    actual = SunSensor(sensor_position, sensor_orientation).observe(sun_position)
    assert_allclose(actual, expected, rtol=1e-02, atol=1e-02)

def test_sun_behind():
    sensor_position = np.array([1,1,1])
    sensor_orientation = np.array([0,0,1,0]) # no rotation
    sun_position = np.array([0,5,0,1])
    sunSensor = SunSensor(sensor_position, sensor_orientation)
    expected = 0
    actual = SunSensor(sensor_position, sensor_orientation).observe(sun_position)
    assert_allclose(actual, expected, rtol=1e-02, atol=1e-02)

def test_sun_45_deg():
    sensor_position = np.array([0,0,0])
    sensor_orientation = np.array([0,0,1,np.pi/2])
    sun_position = np.array([0,5,5,1])
    expected = np.cos(np.pi/4.0)
    actual = SunSensor(sensor_position, sensor_orientation).observe(sun_position)
    sunSensor = SunSensor(sensor_position, sensor_orientation)
    assert_allclose(actual, expected, rtol=1e-02, atol=1e-02)
