import pytest
import math
import numpy as np
from numpy.testing import assert_allclose
import transforms3d as tf
from spaceggs.models import Entity

C30 = np.sqrt(3.0)/2.0
S30 = 0.5
C45 = 0.707106781186547
S45 = 0.707106781186547
C60 = 0.5
S60 = np.sqrt(3.0)/2.0
C90 = 0.0
S90 = 1.0
C360 = 1.0
S360 = 0.0
C380 = 0.939692620785908
S380 = 0.342020143325669
C400 = 0.766044443118978
S400 = 0.642787609686539
C420 = 0.5
S420 = np.sqrt(3.0)/2.0


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
    assert_allclose(actual, expected, atol=1e-07)
	
def test_transformation_matrix_x_90():
    position = np.array([2,3,5])
    orientation = np.array([1,0,0,math.pi/2.0]) 
    expected = np.array([
        [1,0,0,2],
        [0,C90,-S90,3],
        [0,S90,C90,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_y_90():
    position = np.array([2,3,5])
    orientation = np.array([0,1,0,math.pi/2.0]) 
    expected = np.array([
        [C90,0,S90,2],
        [0,1,0,3],
        [-S90,0,C90,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)
    
def test_transformation_matrix_z_90():
    position = np.array([2,3,5])
    orientation = np.array([0,0,1,math.pi/2.0]) 
    expected = np.array([
        [C90,-S90,0,2],
        [S90,C90,0,3],
        [0,0,1,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_x_60():
    position = np.array([2,3,5])
    orientation = np.array([1,0,0,math.pi/3.0]) 
    expected = np.array([
        [1,0,0,2],
        [0,C60,-S60,3],
        [0,S60,C60,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_y_60():
    position = np.array([2,3,5])
    orientation = np.array([0,1,0,math.pi/3.0]) 
    expected = np.array([
        [C60,0,S60,2],
        [0,1.0,0,3],
        [-S60,0,C60,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_z_60():
    position = np.array([2,3,5])
    orientation = np.array([0,0,1,math.pi/3.0]) 
    expected = np.array([
        [C60,-S60,0,2],
        [S60,C60,0,3],
        [0,0,1,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_x_30():
    position = np.array([2,3,5])
    orientation = np.array([1,0,0,math.pi/6.0]) 
    expected = np.array([
        [1,0,0,2],
        [0,C30,-S30,3],
        [0,S30,C30,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_y_30():
    position = np.array([2,3,5])
    orientation = np.array([0,1,0,math.pi/6.0]) 
    expected = np.array([
        [C30,0,S30,2],
        [0,1.0,0,3],
        [-S30,0,C30,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_z_30():
    position = np.array([2,3,5])
    orientation = np.array([0,0,1,math.pi/6.0]) 
    expected = np.array([
        [C30,-S30,0,2],
        [S30,C30,0,3],
        [0,0,1,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)
	
def test_transformation_matrix_x_360():
    position = np.array([2,3,5])
    orientation = np.array([1,0,0,2*math.pi]) 
    expected = np.array([
        [1,0,0,2],
        [0,C360,-S360,3],
        [0,S360,C360,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_y_360():
    position = np.array([2,3,5])
    orientation = np.array([0,1,0,2*math.pi]) 
    expected = np.array([
        [C360,0,S360,2],
        [0,1.0,0,3],
        [-S360,0,C360,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_z_360():
    position = np.array([2,3,5])
    orientation = np.array([0,0,1,2*math.pi]) 
    expected = np.array([
        [C360,-S360,0,2],
        [S360,C360,0,3],
        [0,0,1,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)
	
def test_transformation_matrix_x_420():
    position = np.array([2,3,5])
    orientation = np.array([1,0,0,2*math.pi]) 
    expected = np.array([
        [1,0,0,2],
        [0,C360,-S360,3],
        [0,S360,C360,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_y_420():
    position = np.array([2,3,5])
    orientation = np.array([0,1,0,2*math.pi]) 
    expected = np.array([
        [C360,0,S360,2],
        [0,1.0,0,3],
        [-S360,0,C360,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_z_420():
    position = np.array([2,3,5])
    orientation = np.array([0,0,1,2*math.pi]) 
    expected = np.array([
        [C360,-S360,0,2],
        [S360,C360,0,3],
        [0,0,1,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)
	
def test_transformation_matrix_x_420():
    position = np.array([2,3,5])
    orientation = np.array([1,0,0,2*math.pi]) 
    expected = np.array([
        [1,0,0,2],
        [0,C360,-S360,3],
        [0,S360,C360,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-07)

def test_transformation_matrix_y_420():
    position = np.array([2,3,5])
    orientation = np.array([0,1,0,2*math.pi]) 
    expected = np.array([
        [C360,0,S360,2],
        [0,1.0,0,3],
        [-S360,0,C360,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-04)

def test_transformation_matrix_x_90_y_60():
    position = np.array([2,3,5])
    orientation = np.array([0.77459667,0.447213595,0.447213595,1.8234769]) 
    expected = np.array([
        [C60,0,S60,2],
        [S90*S60,C90,-C60*S90,3],
        [-C90*S60,S90,C90*C60,5],
        [0,0,0,1]
    ])
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-04)
	
def test_transformation_matrix_x_30_y_45_z_60():
    position = np.array([2,3,5])
    orientation = np.array([0.5675523978,0.290452662,0.7704034832,1.5244035316])
    expected = np.array([
        [C45*C60,-C45*S60,S45,2],
        [C30*S60+C60*S30*S45,C30*C60-S30*S45*S60,-C45*S30,3],
        [S30*S60-C30*C60*S45,C60*S30+C30*S45*S60,C30*C45,5],
        [0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=1e-04)

def test_transformation_matrix_x_360_y_380_z_400():
    C1 = C360
    position = np.array([2,3,5])
    orientation = np.array([0.156725047300549,0.430598528521216,0.888831911434330,0.777325823456587])
    C1 = C360
    S1 = S360
    C2 = C380
    S2 = S380
    C3 = C400
    S3 = C400
    expected = np.array([
        [C2*C3,-C2*S3,S2,2],
        [C1*S3+C3*S1*S2,C1*C3-S1*S2*S3,-C2*S1,3],
        [S1*S3-C1*C3*S2,C3*S1+C1*S2*S3,C1*C2,5],
		[0,0,0,1]
    ])    
    actual = Entity(position, orientation).M    
    assert_allclose(actual, expected, atol=0.15)