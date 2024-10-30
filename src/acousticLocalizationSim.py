import math
import numpy as np
from numpy.random import randn

# Acoustic simulation
class AcousticLocalizationSim:
    """ Simulates the acoustic localization algorithm.
    """

    def __init__(self, pos=(0, 0), noise_std=1., noisy = False):
        self.pos = [pos[0], pos[1]]
        self.noise_std = noise_std
        self.estimated_position = [0., 0.]
        self.noisy = noisy

    def get_atan(self, s1, s2):

        dx = s2[0] - s1[0]
        dy = s2[1] - s1[1]

        angle = 0

        if dx > 0:
            if dy > 0:
                angle = math.atan(dy / dx)
            elif dy < 0:
                angle = 2 * math.pi + math.atan(dy / dx)
        elif dx < 0:
            if dy > 0:
                angle = math.pi + math.atan(dy / dx)
            elif dy < 0:
                angle = math.pi + math.atan(dy / dx)
        elif dx == 0:
            if dy > 0:
                angle = math.pi / 2
            elif dy < 0:
                angle = 3 * math.pi / 2
            elif dy == 0:
                angle = 0
        return angle

    def get_relativeAngle(self, robot):
        """ Returns the relative angle between the robot and the speaker
        @param robot: robot position
        @param noise: whether to add noise to the measurement
        @return angle between the robot and the speaker
        """
        angle = math.atan2(self.pos[1] - robot[1], self.pos[0]-robot[0]) #self.get_atan(self.pos, robot)

        #if angle < 0:
            #angle = 2 * math.pi + angle

        if self.noisy:
            return angle + np.random.normal(0, self.noise_std)
        return angle

    def set_noise_std(self, noise_std):
        """ Sets the noise standard deviation
        @param noise_std: standard deviation of the noise
        """
        self.noise_std = noise_std

    def get_position(self):
        """ Returns the position of the speaker
        @param noise: whether to add noise to the measurement
        @return position of the speaker
        """
        if self.noisy:
            return [self.pos[0] + randn() * self.noise_std, self.pos[1] + randn() * self.noise_std]
        return self.pos

    def estimate_position(self, r0, r1):
        """ Returns the estimated position of the speaker
        @param robot: two robot positions
        @param noise: whether to add noise to the measurement
        @return estimated position of the speaker
        """

        th0 = self.get_relativeAngle(r0)
        th1 = self.get_relativeAngle(r1)

        #self.get_atan(r0,r1)
        alpha =  math.atan2(r1[1] - r0[1], r1[0] - r0[0])

        #if alpha < 0:
            #alpha = 2 * math.pi + alpha

        #print('alpha:', alpha*(180/math.pi),'respect to r0', r0, 'y r1:', r1)

        D = math.sqrt((r0[0] - r1[0])**2 + (r0[1] - r1[1])**2) *(np.sin(np.pi/2 + (th0 - alpha)%(np.pi/2))/np.sin(- th0 + th1))

        self.estimated_position[0] = r1[0] + D * np.cos(th1)
        self.estimated_position[1] = r1[1] + D * np.sin(th1)
        return self.estimated_position