import numpy as np

class Robot:
    """ Simulates a ground robot
    """

    def __init__(self, pos, dt, R, th = 0):
        self.d = None
        self.k = None
        self.pos = np.array([pos[0], pos[1]])
        self.dt = dt
        self.R = R
        self.th = th
        self.wheelbase = 1

    def set_tunning(self, k, d):
        """ Sets the tunning parameters
        """
        self.k = k
        self.d = d

    def move(self, x, u):
        hdg = x[2]
        vel = u[0]
        steering_angle = u[1]
        dist = vel * self.dt

        if abs(steering_angle) > 0.001:  # is robot turning?
            beta = (dist / self.wheelbase) * np.tan(steering_angle)
            r = self.wheelbase / np.tan(steering_angle)  # radius

            dx = np.array([-r * np.sin(hdg) + r * np.sin(hdg + beta),
                           r * np.cos(hdg) - r * np.cos(hdg + beta),
                           beta])
        else:  # moving in straight line
            dx = np.array([dist * np.cos(hdg),
                           dist * np.sin(hdg),
                           0])
        return x + dx