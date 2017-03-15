import numpy as np

class KF:
    def __init__(self, n, m, k):
        '''
        Inputs:
        - _n: number of observed and hidden state variables
        - _m: number of control inputs
        - _k: number sensors
        '''
        self._n = n
        self._m = m
        self._k = k
        
        '''
        - x: state estimate, an [n, 1] numpy array where n is the number of observed and hidden state variables
        - F: state transition model, [n, n] numpy array where n is the number of observed and hidden state variables
        - u control input, an [m, 1] numpy array where m is the number of control inputs
        - B: control-input model [n, m] numpy array where n is the number of states variables and m is the number of control inputs
        - P: predicted state estimate covariance, an [n, n] numpy array where n is the number of state variables
        - Q: process noise, an [n, n] numpy array where n is the number of state variables
        - H: observation model, an [k, n] numpy array where k is the number of senesors and n is the number of state variables
        - R: observation noise covariance, an [k, k] numpy array where k is the number of sensors
        
        Note:
        - F, B, Q, H, R are constants
        '''
        self._x = np.zeros((self._n, 1)) # state estimate
        self._F = np.zeros((self._n, self._n)) # state transition model
        self._u = np.zeros((self._m, 1)) # control input
        self._B = np.zeros((self._n, self._m)) # control-input model
        self._P = np.zeros((self._n, self._n)) # predicted state estimate covariance
        self._Q = np.zeros((self._n, self._n)) # process noise
        self._H = np.zeros((self._k, self._n)) # observation model
        self._R = np.zeros((self._k, self._k)) # noise covariance    
    
    def initialize(self, **kwargs):
        '''
        Inputs:
        - kwargs: a dictionary of attributes and values to initialize the filter
        '''
        for key in kwargs:
            if hasattr(self, key):
                attr = getattr(self, key)
                if attr.shape == kwargs[key].shape:
                    attr[:] = kwargs[key]
                else:
                    print("invalid shape for {}: expected {}, and received {}".format(
                            key, attr.shape, kwargs[key].shape))
            else:
                print("unkown attribute: {}".format(key))
        
    
    def predict(self):
        self._x = np.dot(self._F, self._x) + np.dot(self._B, self._u) # predicted state estimate
        self._P = self._F.dot(self._P).dot(self._F.T) + self._Q # predicted state estimate covariance
        
    def update(self, z):
        '''
        Inputs:
        - z: observation, an [k, 1] numpy array where k is the number of sensors
        '''
        y = z - np.dot(self._H, self._x)  # innovation or measurement residual
        S = self._H.dot(self._P).dot(self._H.T) + self._R # innovation (or residual) covariance
        K = self._P.dot(self._H.T).dot(np.linalg.pinv(S)) # kalman gain
        self._x += np.dot(K, y) # Updated (a posteriori) state estimate
        self._P = (np.eye(self._n) - K.dot(self._H)).dot(self._P) # Updated (a posteriori) estimate covariance
        
    def __str__(self):
        return "x: {}, var: {}".format(self._x, np.diagonal(self._P))
    def __rper__(self):
        return self.__str__()
    def get_state(self):
        return np.copy(self._x)