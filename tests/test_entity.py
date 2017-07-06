import pytest

import math
import numpy as np
from numpy.testing import assert_allclose

import r"C:\Python27\Lib\site-packages\transforms3d" as tf

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
	
def test_transformation_matrix_x_90():
    position = np.array([2,3,5])
    orientation = np.array([1,0,0,math.pi/2]) # no rotation
    expected = np.array([
        [1,0,0,2],
        [0,0,-1,3],
        [0,1,0,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, rtol=1e-07)
	
def test_transformation_matrix_y_90():
    position = np.array([2,3,5])
    orientation = np.array([0,1,0,math.pi/2]) # no rotation
    expected = np.array([
        [0,0,1,2],
        [0,1,0,3],
        [-1,0,0,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, rtol=1e-07)
	
def test_transformation_matrix_z_90():
    position = np.array([2,3,5])
    orientation = np.array([0,0,1,math.pi/2]) # no rotation
    expected = np.array([
        [0,-1,0,2],
        [1,0,0,3],
        [0,0,1,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, rtol=1e-07)

def test_transformation_matrix_x_60():
    position = np.array([2,3,5])
    orientation = np.array([0,0,1,math.pi/2]) # no rotation
    expected = np.array([
        [1,0,0,2],
        [0,1/2,-sqrt(3)/2,3],
        [0,sqrt(3)/2,1/2,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, rtol=1e-07)

def test_transformation_matrix_y_60():
    position = np.array([2,3,5])
    orientation = np.array([0,0,1,math.pi/2]) # no rotation
    expected = np.array([
        [1/2,0,sqrt(3)/2,2],
        [0,0,0,3],
        [-sqrt(3)/2,0,1/2,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, rtol=1e-07)

def test_transformation_matrix_z_60():
    position = np.array([2,3,5])
    orientation = np.array([0,0,1,math.pi/2]) # no rotation
    expected = np.array([
        [1/2,-sqrt(3)/2,0,2],
        [sqrt(3)/2,1/2,0,3],
        [0,0,1,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, rtol=1e-07)