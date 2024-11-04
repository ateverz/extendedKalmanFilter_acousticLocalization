## add src to the path
import sys

sys.path.append('src')

# Import needed packages
import simulator as sim
import numpy as np

robot = sim.Robot(robot = (18, -18), d = 10, speaker = (-10, 10), noise_std = 0.1, noisy = True, dim = (2,2))
kp = 0.1
robot.kfSettings(R = kp*np.pi, Q = 0, P = kp*np.pi, F = 1, H = 1)

# General settings
'''

speaker_real_position = (-10, 10)
speaker = sim.Acoustic(pos = speaker_real_position, noise_std = .1, noisy = True)
robot = sim.Robot(x0 = (18, -18), d = 10)
estimator = sim.Estimator(dim = (2,2))


'''
results = sim.Results()



# Main loop
def main():
    robot.main(100)

# Plot the results
def plot():
    results.plot1(robot)

if "__main__" == __name__:
    main()
    plot()