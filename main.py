import numpy as np
import importlib
import matplotlib.pyplot as plt

import maze
import utils
import policies
import gptd

importlib.reload(maze)
importlib.reload(utils)
importlib.reload(gptd)


# ############# BUILD MAZE #############################################################################################
env = maze.Maze(1, 1, 0.1, 0.03, 0.01)

barrier1 = (-0.1, 0.4, -0.75, 0.8)
barrier2 = (0.55, 0.7, -3, 2.525)
barrier3 = (0.7, 1.1, -0.5, 0.6)
barrier4 = (0.2, 0.55, 0.5, 0.1)

env.add_barrier(barrier1[0], barrier1[1], barrier1[2], barrier1[3])
# env.add_barrier(barrier2[0], barrier2[1], barrier2[2], barrier2[3])
# env.add_barrier(barrier3[0], barrier3[1], barrier3[2], barrier3[3])
# env.add_barrier(barrier4[0], barrier4[1], barrier4[2], barrier4[3])


goal1 = (0, 0, 1, 0.05)
env.add_goal(goal1[0], goal1[1], goal1[2], goal1[3])


# ########### DEFINE POLICY ############################################################################################
policy = policies.SimplePolicy([0, 0, 0, 0, 0.8, 0, 0, 0])



# Simulate trajectories
N = 1
T = 1000
trajs = utils.collect_trajectories(env, policy, T, N)

x = trajs[0]["states"]
r = trajs[0]["rewards"]


# Plot maze and trajectory
test_traj = trajs[0]
moves_x = [t[0] for t in test_traj["states"]]
moves_y = [t[1] for t in test_traj["states"]]
env.plot()
plt.plot(moves_x, moves_y, color="C3", marker="o")


# Test GPTD
kernel = gptd.Gaussian_Kernel(sigma=0.1)

# Parameters
gamma = 0.95
nu = 0.05
sigma0 = 1

# Initialization (t=1)
# We stack the xs as columns
xdict = np.array(x[0]).reshape((2, 1))
alpha = np.array([0])
C = np.array([[0]])
Kinv = np.array([[1 / kernel.k(x[0], x[0])]])


# First step (t=2)
# Compute kernel related quantities
ktt = kernel.k(x[1], x[1])
kt = gptd.compute_k(xdict, x[1], kernel)
# In this case (first add to dictionnary) this is equal to ktt
ktminus1 = gptd.compute_k(xdict, x[0], kernel)
# Compute epsilon for test
a = gptd.compute_a(kt, Kinv)
eps = gptd.compute_epsilon(ktt, kt, a)
# Force add to dictionnary
xdict = np.concatenate((xdict, x[1].reshape((2, 1))), axis=1)
Kinv = gptd.augment_Kinv(eps, a, Kinv)
# We initialize H instead of updating it for the first augmentation step
H = gptd.initialize_H(gamma)
# We intialize Q instead of updating it for the first augmentation step
# For that we need to create the kernel matrix
# TODO: is this correct ? Formula imply that the shape of Qt should be (t - 1, t - 1)
K = gptd.compute_K(xdict, kernel)
Q = gptd.intialize_Q(K, H, sigma0)
# At first step we compute alpha directly since we cannot define the recursions
# TODO: Should we initialize r this way (with reward in initialize state equal to -1) ?
alpha = gptd.initialize_alpha(H, Q, np.array([-1, r[0]]))


# Second step






















