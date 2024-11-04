## add src to the path
import sys
sys.path.append('src')

import simulator
import matplotlib.pyplot as plt
import numpy as np
import math
from filterpy.kalman import KalmanFilter
from scipy import stats

# Speaker settings
speaker_real_position = (-10, 10)
noise_std = math.pi/10
speaker = simulator.Acoustic(pos=speaker_real_position, noise_std = noise_std, noisy=True)

## Estimated position
xs = []
ys = []

## Filtered position based on estimation
xsF = []
ysF = []

# Robot settings
xr = [10]
yr = [-18]

robot = simulator.Robot(x0 = xr[0], y0 = yr[0], d =8)

## Calculate first step according to the angle between robot and speaker
th = speaker.angleRespectTo([xr[0], yr[0]])
sigma = robot.move(th)


xr.append(sigma[0])
yr.append(sigma[1])

r0 = np.array([xr[-2], yr[-2]])
r1 = np.array([xr[-1], yr[-1]])

## Compute first estimated position of speaker
S = speaker.estimate_position(r0, r1)

## Estimated position history
xs.append(S[0])
ys.append(S[1])

## Kalman filter settings
kf = KalmanFilter(dim_x=2, dim_z=2)
kf.x = np.array(S)
kp = 0.1
kf.R *= kp * noise_std * np.eye(2)
kf.Q *= 0 #noise_std * np.eye(2)
kf.P *= kp * noise_std * np.eye(2)
kf.F = np.eye(2)
kf.H = np.eye(2)

## Filtered position history
kf.predict()
kf.update(S)
xsF.append(kf.x[0])
ysF.append(kf.x[1])

k = 0

check = 5000

th = speaker.angleRespectTo([xr[-1], yr[-1]])

for i in range(100):
    sigma = robot.move(th)

    xr.append(sigma[0])
    yr.append(sigma[1])

    ## Compute estimated position
    S = speaker.estimate_position(r0, r1)

    ## Estimated position history
    xs.append(S[0])
    ys.append(S[1])

    kf.predict()
    kf.update(S)

    xsF.append(kf.x[0])
    ysF.append(kf.x[1])

    r0 = np.array([xr[-2], yr[-2]])
    r1 = np.array([xr[-1], yr[-1]])

    ## Compute angle between robot and speaker
    th = speaker.angleRespectTo(r1)

    k += 1

    if math.sqrt((xsF[-1] - speaker_real_position[0])**2 + (ysF[-1] - speaker_real_position[1])**2) <= 1:
        print('Converged at iteration: ', k)
        break

print('Iterations done: ', k)

fig = plt.figure()


ax1 = plt.subplot2grid((2,7), (0,0), colspan=4, rowspan=2)
ax2 = plt.subplot2grid((2,7), (0,4), colspan=3)
ax3 = plt.subplot2grid((2,7), (1,4), colspan=3)

alpha = np.linspace(0.1,0,np.size(xr))

ax1.plot(xr, yr, 'g--', label='Robot')

for i in range(np.size(xs)):
    ax1.plot(xs[i], ys[i], 'ro', alpha=alpha[i]*2)
    ax1.plot(xsF[i], ysF[i], 'ko', alpha=alpha[i])

ax1.plot(speaker_real_position[0],speaker_real_position[1], 'bo', label='Speaker')
ax1.plot(xr[0], yr[0], 'go')
ax1.plot(np.mean(xs), np.mean(ys), 'ro', label = 'Mean measurement')
ax1.plot(stats.mode(xs), stats.mode(ys), 'yo', label = 'Mode measurement')
ax1.plot(xsF[-1], ysF[-1], 'ko')
#ax1.legend(loc='best')
ax1.set_ylim(-20, 20)
ax1.set_xlim(-20, 20)

ax2.plot(range(np.size(xs)), xs, 'r', label='Localization method')
ax3.plot(range(np.size(ys)), ys, 'r')
ax2.axhline(y = speaker_real_position[0], color = 'b')
ax3.axhline(y = speaker_real_position[1], color = 'b')
ax2.plot(range(np.size(xsF)), xsF, 'k', label='Kalman filter')
ax3.plot(range(np.size(ysF)), ysF, 'k')
#ax2.set_ylim(-20, 20)
#ax3.set_ylim(-20, 20)
#ax2.legend(['Estimated position', 'Filtered position', 'Real position'], loc='best')
#ax3.legend(['Estimated position', 'Filtered position', 'Real position'], loc='best')
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

folder = '/home/ateveraz/Documents/phd/projects/extendedKalmanFilter_acousticLocalization/doc/images/'
plt.savefig(folder + 'simulator_filter.pdf', format='pdf', dpi=300)

plt.show()

print('Real position:', speaker_real_position)
print('Filtered position (KF):', [xsF[-1], ysF[-1]])
print('Estimated position:', [xs[-1], ys[-1]])