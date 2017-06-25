import numpy as np
from .entity import Entity

class Sensor(Entity):
    def __init__(self, position, orientation):
        '''Position and Orientation are with respect to the Satellite's coordinate system'''
        Entity.__init__(self, position, orientation)
        
    def observe(self):
        raise 'You should override this method'

class SunSensor(Sensor):
    def __init__(self, position, orientation):
        '''
        Inputs:
        - head: A numpy array of shape [4, ] representing the sensor's heading in Homogeneous coordinates
        '''
        Sensor.__init__(self, position, orientation)
        
    def observe(self, sun):
        '''
        Inputs:
        - sun: A numpy array of shape [4, ] representing the position of the Sun with respect
            to the satellite in Homogeneous coordinates
            
        Returns:
        - A number between 0 and 1
        '''
        
        sun = self.M.dot(np.asarray(sun)) # Mapping to sensor's coordinate system
        a, b = sun[:3], self.head[:3]
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b) # normalize the heading and sun vectors
        return np.dot(a, b)