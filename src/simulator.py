import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.random import randn
from filterpy.kalman import KalmanFilter


# Acoustic simulation
class Acoustic:
    """ Simulates the acoustic localization algorithm.
    """
    def __init__(self, pos=(0, 0), noise_std=1., noisy = False):
        self.positionSpeaker = pos
        self.noise_std = noise_std*np.pi
        self.noisy = noisy

    def angleRespectTo(self, robot):
        """ Returns the relative angle between the robot and the speaker
        @param robot: robot position
        @param noise: whether to add noise to the measurement
        @return angle between the robot and the speaker
        """

        angle = math.atan2(self.positionSpeaker[1] - robot[1], self.positionSpeaker[0] - robot[0])

        if self.noisy:
            return angle + np.random.normal(0, self.noise_std)
        return angle

class Estimator:
    def __init__(self, dim = (2,2)):
        self.dim = dim
        self.kf = KalmanFilter(dim_x = dim[0], dim_z = dim[1])
        self.localized = np.array([])
        self.filtered = np.array([])

    def settings(self, R, Q, P, F, H):
        self.kf.R *= R * np.eye(self.dim[0])
        self.kf.Q *= Q * np.eye(self.dim[0])
        self.kf.P *= P * np.eye(self.dim[0])
        self.kf.F  = F * np.eye(self.dim[0])
        self.kf.H  = H * np.eye(self.dim[0])

    def setInitial(self, x0):
        self.kf.x = x0
        self.filtered = x0

    def filter(self, z):
        self.kf.predict()
        self.kf.update(z)
        self.filtered = np.vstack((self.filtered, self.kf.x))
        return self.kf.x

    def angleCorrection(self, th, robot):
        x = self.kf.x
        angle = math.atan2(x[1] - robot[1], x[0] - robot[0])
        return (angle - th) % (2*np.pi)


    def localize(self, th0, th1, r0, r1):
        """ Returns the estimated position of the speaker
        @param robot: two robot positions
        @param noise: whether to add noise to the measurement
        @return estimated position of the speaker
        """

        alpha =  math.atan2(r1[1] - r0[1], r1[0] - r0[0])

        D = math.sqrt((r0[0] - r1[0])**2 + (r0[1] - r1[1])**2) *(np.sin(np.pi/2 + (th0 - alpha)%(np.pi/2))/np.sin(- th0 + th1))

        loc = r1 + D * np.array([np.cos(th1), np.sin(th1)])

        if self.localized.size == 0:
            self.localized = loc
        else:
            self.localized = np.vstack((self.localized, loc))
        return loc

class Robot(Acoustic, Estimator):
    def __init__(self, robot = None, d = 0.1, speaker = (0,0), noise_std = 0.1, noisy = False, dim = (2,2)):
        if robot is None:
            robot = [0., 0.]
        self.x = np.array([robot])
        self.d = d
        self.th0 = 0
        self.th1 = 0

        Acoustic.__init__(self, pos = speaker, noise_std = noise_std, noisy = noisy)
        Estimator.__init__(self, dim = dim)

    def kfSettings(self, R, Q, P, F, H):
        self.settings(R, Q, P, F, H)

    def kinematics(self, th):
        return np.array([np.cos(th) + np.sin(th), np.sin(th) - np.cos(th)])

    def move(self, th):
        x_ = self.x[-1] + self.d * self.kinematics(th)
        self.x = np.vstack((self.x, x_))

    def firstMove(self):
        self.th0 = self.angleRespectTo(self.x[0, :])
        self.move(self.th0)
        self.th1 = self.angleRespectTo(self.x[1, :])

        loc = self.localize(self.th0, self.th1, self.x[0, :], self.x[1,:])
        self.setInitial(loc)
        self.filter(loc)

    def main(self, n = 100):
        self.firstMove()
        for i in range(n):
            if np.linalg.norm(self.kf.P) < 0.003:
                self.th0 = self.angleCorrection(self.th1, self.x[-1])
            else:
                self.th0 = self.th1
            self.move(self.th0)
            self.th1 = self.angleRespectTo(self.x[-1, :])
            loc = self.localize(self.th0, self.th1, self.x[-2, :], self.x[-1, :])
            self.filter(loc)
            cov = np.linalg.norm(self.kf.P)
            print('Covarianza de la aproximación: ', cov)
            print('Error de la aproximación: ', np.linalg.norm(loc - self.positionSpeaker))




class Results:
    def __init__(self):
        pass

    def plot1(self, robot):
        fig = plt.figure()

        ax1 = plt.subplot2grid((2, 7), (0, 0), colspan=4, rowspan=2)
        ax2 = plt.subplot2grid((2, 7), (0, 4), colspan=3)
        ax3 = plt.subplot2grid((2, 7), (1, 4), colspan=3)

        alpha = np.linspace(0.1, 0, np.size(robot.x[:, 0]))

        ax1.plot(robot.x[:, 0], robot.x[:, 1], 'g--', label='Robot')

        for i in range(np.size(robot.localized[:, 0])):
            ax1.plot(robot.localized[i, 0], robot.localized[i, 1], 'ro', alpha=alpha[i] * 2)
            ax1.plot(robot.filtered[i, 0], robot.filtered[i, 1], 'ko', alpha=alpha[i])

        ax1.plot(robot.positionSpeaker[0], robot.positionSpeaker[1], 'bo', label='Speaker')
        ax1.plot(robot.x[0, 0], robot.x[0, 1], 'go')
        ax1.plot(np.mean(robot.filtered[:, 0]), np.mean(robot.filtered[:, 1]), 'ro', label='Mean measurement')
        ax1.plot(stats.mode(robot.filtered[:, 0]), stats.mode(robot.filtered[:, 1]), 'yo',
                 label='Mode measurement')
        ax1.plot(robot.filtered[-1, 0], robot.filtered[-1, 1], 'ko')
        # ax1.legend(loc='best')
        ax1.set_ylim(-20, 20)
        ax1.set_xlim(-20, 20)

        ax2.plot(range(np.size(robot.localized[:, 0])), robot.localized[:, 0], 'r', label='Localization method')
        ax3.plot(range(np.size(robot.localized[:, 1])), robot.localized[:, 1], 'r')
        ax2.axhline(y=robot.positionSpeaker[0], color='b')
        ax3.axhline(y=robot.positionSpeaker[1], color='b')
        ax2.plot(range(np.size(robot.filtered[:, 0])), robot.filtered[:, 0], 'k', label='Kalman filter')
        ax3.plot(range(np.size(robot.filtered[:, 1])), robot.filtered[:, 1], 'k')
        # ax2.set_ylim(-20, 20)
        # ax3.set_ylim(-20, 20)
        # ax2.legend(['Estimated position', 'Filtered position', 'Real position'], loc='best')
        # ax3.legend(['Estimated position', 'Filtered position', 'Real position'], loc='best')
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('x coordinate [m]')
        ax3.set_ylabel('y coordinate [m]')
        ax1.set_xlabel('x coordinate [m]')
        ax1.set_ylabel('y coordinate [m]')
        ax3.set_xlabel('Samples')

        labels = []
        handles = []
        for sbp in fig.axes:
            handles_, label_ = sbp.get_legend_handles_labels()
            labels.extend(label_)
            handles.extend(handles_)

        plt.tight_layout()

        fig.legend(handles, labels, loc='upper right', ncol=len(handles), prop={'size': 6})
        plt.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.1)

        # folder = '/home/ateveraz/Documents/phd/projects/extendedKalmanFilter_acousticLocalization/doc/images/'
        # plt.savefig(folder + 'simulator_filter.pdf', format='pdf', dpi=300)

        plt.show()