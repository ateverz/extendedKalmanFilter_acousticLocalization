import acousticLocalizationSim as ac
import matplotlib.pyplot as plt
import numpy as np
import math
import robot as rb
from filterpy.kalman import KalmanFilter

noise_std = 0.01
speaker = AcousticLocalizationSim(pos=(5, 5), noise_std = noise_std)

xd = [-1, -1]

robot = rb.Robot(xd, 0.1, 0.1)

angle = [speaker.get_relativeAngle(xd, noise=True)]

x = []
y = []
xr = []
yr = []

xrF = []
yrF = []

d = 0.1

kf = KalmanFilter(dim_x=2, dim_z=2)
kf.x = np.array(xd)
kf.R *= noise_std * np.eye(2)
kf.Q *= noise_std * np.eye(2)
kf.P *= noise_std * np.eye(2)
kf.F = np.eye(2)
kf.H = np.eye(2)

th = angle[0]

X = np.array([xd[0], xd[1], 0])

for i in range(500):
    coord = [X[0], X[1]]

    th = speaker.get_relativeAngle(coord, noise=True)
    angle.append(th)

    xd[0] += (math.cos(th) + math.sin(th)) * d
    xd[1] += (math.sin(th) - math.cos(th)) * d
    pos = speaker.get_position(noise=False)

    kf.predict()
    kf.update(xd)

    xrF.append(kf.x[0])
    yrF.append(kf.x[1])

    beta = - 10 * (X[2] - th)
    v = - 10 * math.sqrt((X[0] - kf.x[0])**2 + (X[1] - kf.x[1])**2) * math.cos(X[2] - th)
    X = robot.move(X, [v, beta])

    xr.append(X[0])
    yr.append(X[1])

    #xd[0] = - 1 * (xd[0] - kf.x[0])
    #xd[1] = - 1 * (xd[1] - kf.x[1])

    x.append(pos[0])
    y.append(pos[1])
    xr.append(xd[0])
    yr.append(xd[1])
    if math.sqrt((xd[0] - pos[0])**2 + (xd[1] - pos[1])**2) < 0.001:
        break


plt.plot(angle, 'ro')
plt.title('Estimated angle')
plt.xlabel('Samples')
plt.ylabel('Angle')
plt.show()

plt.plot(x[0],y[0], 'bo')
plt.plot(xr[0], yr[0], 'go')
plt.plot(xr, yr, 'rx')
plt.plot(xrF, yrF, 'k')
plt.title('Estimation of position')
plt.xlabel('X-coord')
plt.ylabel('Y-coord')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()