import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.random import randn
from filterpy.kalman import KalmanFilter
from matplotlib.colors import colorConverter

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
        self.kf.F = F * np.eye(self.dim[0])
        self.kf.H = H * np.eye(self.dim[0])

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
        return (angle + th)/2 # %np.pi # /2


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
    def __init__(self, robot = None, d = 0.1, speaker = (0,0), noise_std = 0.1, noisy = False, to_filter = None):
        if robot is None:
            robot = [0., 0.]
        self.x = np.array([robot])
        self.d = d
        self.th0 = 0
        self.th1 = 0
        self.error = np.array([])
        self.counter = 0
        self.radius = 1

        if to_filter is None:
            raise SyntaxError('Filter is not defined')
        else:
            self.toFilter = to_filter
            if self.toFilter == 'cartesian':
                dim = (2, 2)
            elif self.toFilter == 'angular':
                dim = (1, 1)
            elif self.toFilter == 'none':
                dim = (1, 1)
            else:
                raise SyntaxError('Select a valid filter')


        Acoustic.__init__(self, pos = speaker, noise_std = noise_std, noisy = noisy)
        Estimator.__init__(self, dim = dim)

    def kfSettings(self, R, Q, P, F, H):
        self.settings(R, Q, P, F, H)

    def kinematics(self, th):
        return np.array([np.cos(th)+np.sin(th), np.sin(th)-np.cos(th)])

    def move(self, th):
        x_ = self.x[-1,:] + self.d * self.kinematics(th)
        self.x = np.vstack((self.x, x_))

    def classic(self):
        self.th0 = self.th1
        self.move(self.th0)
        self.th1 = self.angleRespectTo(self.x[-1, :])
        self.localize(self.th0, self.th1, self.x[-2, :], self.x[-1, :])

    def cartesianFilter(self):
        if np.std(self.filtered[-5:]) < 0.1:
            self.th0 = self.angleCorrection(self.th1, self.x[-1])
        else:
            self.th0 = self.th1
        self.move(self.th0)
        self.th1 = self.angleRespectTo(self.x[-1, :])
        loc = self.localize(self.th0, self.th1, self.x[-2, :], self.x[-1, :])
        self.filter(loc)

    def angularFilter(self):
        if np.std(self.filtered[-5:]) < 0.05:
            self.th0 = (np.mean(self.kf.x) + self.th1)/2 # %np.pi # /2
        else:
            self.th0 = self.th1
        self.move(self.th0)
        self.th1 = self.angleRespectTo(self.x[-1, :])
        self.localize(self.th0, self.th1, self.x[-2, :], self.x[-1, :])
        self.filter(self.th1)

    def firstMove(self):
        self.th0 = self.angleRespectTo(self.x[0, :])
        self.move(self.th0)
        self.th1 = self.angleRespectTo(self.x[1, :])

        loc = self.localize(self.th0, self.th1, self.x[0, :], self.x[1,:])
        self.error = np.array([np.linalg.norm(self.positionSpeaker - loc)])

        if self.toFilter == 'cartesian':
            self.setInitial(loc)
            self.filter(loc)
        elif self.toFilter == 'angular':
            self.setInitial(self.th0)
            self.filter(self.th1)
        elif self.toFilter == 'none':
            pass
        else:
            raise SyntaxError('Filter is not defined')

    def main(self, n = 100):
        self.firstMove()
        for i in range(n):
            if self.toFilter == 'cartesian':
                self.cartesianFilter()
            elif self.toFilter == 'angular':
                self.angularFilter()
            elif self.toFilter == 'none':
                self.classic()
            else:
                raise SyntaxError('Filter is not defined')
            self.error = np.append(self.error, np.linalg.norm(self.positionSpeaker - self.localized[-1, :]))
            self.counter += 1
            mean = np.mean(self.localized[-5:], axis=0)
            if np.linalg.norm(mean - self.localized[-1,:]) < 0.3:
                break
            if np.linalg.norm(self.positionSpeaker - self.x[-1, :]) <= self.radius:
                break
            #if np.mean(self.error[-5:]) < 0.5 :
            #    break

    def showStadistics(self):
        s = 5
        print(f'::::::::::::: Filter applied: {self.toFilter} ( Iterations: {self.counter}) ::::::::::::: ')
        print(
            f'Mean: {np.mean(self.error[-s:])} Max: {np.max(self.error[-s:])} Min: {np.min(self.error[-s:])} Std: {np.std(self.error[-s:])}')
        print(f'Estimated position: {np.mean(self.localized[-5:], axis=0)} \n')


class Results:
    def __init__(self, folder):
        self.folder = folder

    def plot1(self, robot):
        fig = plt.figure()

        ax1 = plt.subplot2grid((2, 7), (0, 0), colspan=4, rowspan=2)
        ax2 = plt.subplot2grid((2, 7), (0, 4), colspan=3)
        ax3 = plt.subplot2grid((2, 7), (1, 4), colspan=3)

        alpha = np.linspace(0.1, 0, np.size(robot.x[:, 0]))

        ax1.plot(robot.x[:, 0], robot.x[:, 1], 'g--', label='Robot')

        for i in range(np.size(robot.localized[:, 0])):
            ax1.plot(robot.localized[i, 0], robot.localized[i, 1], 'ro', alpha=alpha[i] * 2)
            if robot.toFilter == 'cartesian':
                ax1.plot(robot.filtered[i, 0], robot.filtered[i, 1], 'ko', alpha=alpha[i])

        ax1.plot(robot.positionSpeaker[0], robot.positionSpeaker[1], 'bo', label='Speaker')
        ax1.plot(robot.x[0, 0], robot.x[0, 1], 'go')
        if robot.toFilter == 'cartesian':
            ax1.plot(np.mean(robot.filtered[:, 0]), np.mean(robot.filtered[:, 1]), 'ro', label='Mean measurement')
            ax1.plot(stats.mode(robot.filtered[:, 0]), stats.mode(robot.filtered[:, 1]), 'yo', label='Mode measurement')
            ax1.plot(robot.filtered[-1, 0], robot.filtered[-1, 1], 'ko')
        # ax1.legend(loc='best')
        ax1.set_ylim(-20, 20)
        ax1.set_xlim(-20, 20)

        ax2.plot(range(np.size(robot.localized[:, 0])), robot.localized[:, 0], 'r', label='Localization method')
        ax3.plot(range(np.size(robot.localized[:, 1])), robot.localized[:, 1], 'r')
        ax2.axhline(y=robot.positionSpeaker[0], color='b')
        ax3.axhline(y=robot.positionSpeaker[1], color='b')
        if robot.toFilter == 'cartesian':
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

    def plotEstimationPerAxis(self, rNone, rAngular, rCartesian, saved = None):
        fig, ax = plt.subplots(3, 1, figsize=(12, 9))

        t_none = range(rNone.localized[:, 0].size)
        t_ang = range(rAngular.localized[:, 0].size)
        t_cart = range(rCartesian.localized[:, 0].size)

        ax[0].plot(t_none, rNone.localized[:, 0], label='Without filter', color='r')
        ax[0].plot(t_ang, rAngular.localized[:, 0], label='Angular filter', color='b')
        ax[0].plot(t_cart, rCartesian.localized[:, 0], label='Cartesian filter', color='g')
        ax[0].axhline(y=rNone.positionSpeaker[0], color='k')
        ax[0].axvline(x=t_none[-1], color='r', linestyle='--')
        ax[0].axvline(x=t_ang[-1], color='b', linestyle='--')
        ax[0].axvline(x=t_cart[-1], color='g', linestyle='--')
        ax[0].set_ylabel('X coordinate [m]')


        ax[1].plot(t_none, rNone.localized[:, 1], color='r')
        ax[1].plot(t_ang, rAngular.localized[:, 1], color='b')
        ax[1].plot(t_cart, rCartesian.localized[:, 1], color='g')
        ax[1].axhline(y=rNone.positionSpeaker[1], color='k')
        ax[1].set_ylabel('Y coordinate [m]')

        ax[2].plot(t_none, rNone.error, color='r')
        ax[2].plot(t_ang, rAngular.error, color='b')
        ax[2].plot(t_cart, rCartesian.error, color='g')
        ax[2].set_ylabel('Error [m]')

        labels = []
        handles = []
        max_ = np.max([t_none[-1], t_ang[-1], t_cart[-1]])
        for sbp in ax:
            handles_, label_ = sbp.get_legend_handles_labels()
            labels.extend(label_)
            handles.extend(handles_)
            sbp.set_xlabel('Samples')
            sbp.set_xlim(0, max_)
            sbp.axvline(x=t_none[-1], color='r', linestyle='--')
            sbp.axvline(x=t_ang[-1], color='b', linestyle='--')
            sbp.axvline(x=t_cart[-1], color='g', linestyle='--')

        plt.tight_layout()

        fig.legend(handles, labels, loc='upper right', ncol=len(handles), prop={'size': 10})
        plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.05)

        if saved is not None:
            plt.savefig(self.folder + saved, format='pdf', dpi=300)
        plt.show()

    def showTrajectory(self, rNone, rAngular, rCartesian, saved = None):
        est_none = np.mean(rNone.localized[-5:], axis=0)
        est_ang = np.mean(rAngular.localized[-5:], axis=0)
        est_cart = np.mean(rCartesian.localized[-5:], axis=0)

        fig = plt.figure(figsize=(8, 8))
        fc = colorConverter.to_rgba('black', alpha=0.2)
        circle1 = plt.Circle((rNone.positionSpeaker[0], rNone.positionSpeaker[1]), rNone.radius, fc=fc)
        plt.gca().add_patch(circle1)

        plt.plot(rNone.x[:, 0], rNone.x[:, 1], 'r--', label='Without filter')
        plt.plot(rAngular.x[:, 0], rAngular.x[:, 1], 'b--', label='Angular filter')
        plt.plot(rCartesian.x[:, 0], rCartesian.x[:, 1], 'g--', label='Cartesian filter')
        plt.scatter(est_none[0], est_none[1], s = 120, marker='+', color = 'r')
        plt.scatter(est_ang[0], est_ang[1], s = 120, marker='+', color = 'b')
        plt.scatter(est_cart[0], est_cart[1], s = 120, marker='+', color = 'g')
        plt.plot(rNone.x[0, 0], rNone.x[0, 1], 'ko', label='Robot at start')
        plt.scatter(rNone.x[0, 0], rNone.x[0, 1], s=120, marker='o', color='k')
        plt.scatter(rNone.positionSpeaker[0], rNone.positionSpeaker[1], s=120, marker='*', color='k', label='Speaker')

        plt.legend(loc='best')
        plt.xlabel('X coordinate [m]')
        plt.ylabel('Y coordinate [m]')
        plt.title('Acoustic localization')

        plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.065)

        if saved is not None:
            plt.savefig(self.folder + saved, format='pdf', dpi=300)

        plt.show()
