## add src to the path
import sys

sys.path.append('src')

# Import needed packages
import simulator as sim
import numpy as np

# Create robots
def createRobot(to_filter):
    robot = sim.Robot(to_filter=to_filter, robot=(18, -18), d=8, speaker=(-10, 10), noise_std=0.05, noisy=True)
    kp = 1
    if to_filter == 'angular':
        robot.kfSettings(R=kp * np.pi, Q= kp * np.pi, P=kp * np.pi, F=1, H=1)
    else:
        robot.kfSettings(R=kp * np.pi, Q= 0, P=kp * np.pi, F=1, H=1)
    return robot

rCartesian = createRobot('cartesian')
rAngular = createRobot('angular')
rNone = createRobot('none')
results = sim.Results('/home/ateveraz/Documents/phd/projects/extendedKalmanFilter_acousticLocalization/doc/images/')


# Main loop
if "__main__" == __name__:
    rNone.main(50)
    rAngular.main(50)
    rCartesian.main(50)

    rNone.showStadistics()
    rAngular.showStadistics()
    rCartesian.showStadistics()

    results.plotEstimationPerAxis(rNone, rAngular, rCartesian, saved = 'estimationPerAxis.pdf')
    results.showTrajectory(rNone, rAngular, rCartesian, saved = 'trajectory.pdf')

    sys.exit()