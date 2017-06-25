import numpy as np
from scipy.linalg import cholesky
from numpy.linalg import inv

'''
This code is adapted from ch-10 of Kalman-and-Bayesian-Filters-in-Python
http://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb#Implementation-of-the-UKF
'''
class MerweScaledSigmaPoints:
    def __init__(self, n, a, b, k):
        l = a ** 2 * (n + k) - n
        Wm = np.full(2 * n + 1, 1. / (2 * (n + l)))
        Wc = np.full(2 * n + 1, 1. / (2 * (n + l)))
        Wm[0] = l / (l + n)
        Wc[0] = l / (l + n) + 1 - a ** 2 + b
        
        self.Wm, self.Wc, self.n, self.l = Wm, Wc, n, l
        self.num_sigmas = 2 * n + 1
                
    def sigma_points(self, X, P):
        sigmas = np.zeros((2 * self.n + 1, self.n))
        U = cholesky((self.n + self.l) * P)
        
        sigmas[0] = X
        for k in range (self.n):
            sigmas[k + 1]   = X + U[k]
            sigmas[self.n + k + 1] = X - U[k]
            
        return sigmas
    
'''
This code is adapted from ch-10 of Kalman-and-Bayesian-Filters-in-Python
http://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb#Implementation-of-the-UKF
'''
class UKF:
    def __init__(self, n,k,dt,f,h,M):
        '''
        Inputs:
        - n: number of observed and hidden state variables
        - k: number of control inputs
        - dt: delta t
        - f: state transition function
        - h: observation model
        - M: a wrapper object for sigma points
        '''
        self._n, self._k, self.dt, self.f, self.h, self.M = n, k, dt, f, h, M
        '''
        - _x: state estimate, an [n, 1] numpy array where n is the number of observed and hidden state variables
        - _P: predicted state estimate covariance, an [n, n] numpy array where n is the number of state variables
        - _Q: process noise, an [n, n] numpy array where n is the number of state variables
        - _R: observation noise covariance, an [k, k] numpy array where k is the number of sensors        
        '''
        self._x = np.zeros(self._n) # state estimate
        self._P = np.identity(self._n) # predicted state estimate covariance
        self._Q = np.identity(self._n) # process noise
        self._R = np.identity(self._k) # noise covariance
        '''
        - sigma_f: a [num_sigma_points, n] numpy array holding sigma points mapped by state transition function
        - sigma_h: a [num_sigma_points, k] numpy array holding sigma points mapped by observation model
        '''
        self.sigmas_f = np.zeros((self.M.num_sigmas, self._n))
        self.sigmas_h = np.zeros((self.M.num_sigmas, self._k))
        
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
        
    def __str__(self):
        return "x: {}, var: {}".format(self._x, np.diagonal(self._P))
    def __rper__(self):
        return self.__str__()
    def get_state(self):
        return np.copy(self._x)
    
    def unscented_transform(self, sigmas, Wm, Wc, Q):
        x = np.dot(Wm, sigmas)
        
        kmax, n = sigmas.shape
        P = np.zeros((n, n))
        for k in range(kmax):
            y = sigmas[k] - x
            P += Wc[k] * np.outer(y, y) 
        P += Q
        
        return x, P
        
    def predict(self):
        sigmas = self.M.sigma_points(self._x, self._P)

        for i in range(self.M.num_sigmas):
            self.sigmas_f[i] = self.f(sigmas[i], self.dt)

        self.xp, self.Pp = self.unscented_transform(self.sigmas_f, self.M.Wm, self.M.Wc, self._Q)
        
    def update(self, z):
        sigmas_f, sigmas_h = self.sigmas_f, self.sigmas_h
        
        for i in range(self.M.num_sigmas):
            sigmas_h[i] = self.h(sigmas_f[i])
        
        zp, Pz = self.unscented_transform(sigmas_h, self.M.Wm, self.M.Wc, self._R)
        
        Pxz = np.zeros((self._n, self._k))
        for i in range(self.M.num_sigmas):
            Pxz += self.M.Wc[i] * np.outer(sigmas_f[i] - self.xp, sigmas_h[i] - zp)

        K = Pxz.dot(inv(Pz)) # Kalman gain

        self._x = self.xp + K.dot(z-zp)
        self._P = self.Pp - K.dot(Pz).dot(K.T)