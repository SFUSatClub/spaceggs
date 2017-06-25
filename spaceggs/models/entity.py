import numpy as np
import transforms3d as tf

class Entity(object):
    def __init__(self, position, orientation):
        '''
        Inputs:
        - position: A numpy array of shape [3, ] representing the position of the object
            in the Cartesian Coordinates System
            
        - orientation: A numpy array of shape [4, ] representing the rotation of the object
            from the x axis in the Cartesian Coordinates System. The first 3 components refer
            to axis and the last component is the rotation
        '''
        self.pos = np.asarray(position)
        self.ori = np.asarray(orientation)
        '''
        By convention all Entities are heading towards X axis
        '''
        self.head = np.array([1, 0, 0, 1])
        
        # M is the coordinate system transformation matrix
        self.M = np.zeros((4, 4))
        self.M[:3, :3] = tf.axangles.axangle2mat(self.ori[:3], self.ori[3])
        self.M[3, 3] = 1
        self.M[:3, 3] = self.pos