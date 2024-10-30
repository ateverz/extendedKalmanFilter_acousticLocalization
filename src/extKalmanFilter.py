import numpy as np

class extKalmanFilter:
    def __init__(self, x0, P0, Q, R, f, h, F, H):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R
        self.f = f
        self.h = h
        self.F = F
        self.H = H

    def predict(self, u):
        self.x = self.f(self.x, u)
        self.P = self.F(self.x, u) @ self.P @ self.F(self.x, u).T + self.Q

    def update(self, z):
        y = z - self.h(self.x)
        S = self.H(self.x) @ self.P @ self.H(self.x).T + self.R
        K = self.P @ self.H(self.x).T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H(self.x) @ self.P

    def estimate(self, u, z):
        self.predict(u)
        self.update(z)

    def getEstimate(self):
        return self.x

    def getErrorCovariance(self):
        return self.P

    def setEstimate(self, x):
        self.x = x

    def setErrorCovariance(self, P):
        self.P = P

    def setProcessNoiseCovariance(self, Q):
        self.Q = Q

    def setMeasurementNoiseCovariance(self, R):
        self.R = R

    def setProcessModel(self, f, F):
        self.f = f
        self.F = F

    def setMeasurementModel(self, h, H):
        self.h = h
        self.H = H

    def getProcessNoiseCovariance(self):
        return self.Q

    def getMeasurementNoiseCovariance(self):
        return self.R

    def getProcessModel(self):
        return self.f, self.F

    def getMeasurementModel(self):
        return self.h, self.H