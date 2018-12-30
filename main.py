import numpy as np
import importlib
import matplotlib.pyplot as plt

import maze
import utils
import policies

importlib.reload(maze)
importlib.reload(utils)


# ############# BUILD MAZE #############################################################################################
env = maze.Maze(1, 1, 0.1, 0.05, 0.01)

barrier1 = (-0.1, 0.4, -0.75, 0.8)
barrier2 = (0.55, 0.7, -3, 2.525)
barrier3 = (0.7, 1.1, -0.5, 0.6)
barrier4 = (0.2, 0.55, 0.5, 0.1)

env.add_barrier(barrier1[0], barrier1[1], barrier1[2], barrier1[3])
env.add_barrier(barrier2[0], barrier2[1], barrier2[2], barrier2[3])
env.add_barrier(barrier3[0], barrier3[1], barrier3[2], barrier3[3])
env.add_barrier(barrier4[0], barrier4[1], barrier4[2], barrier4[3])


goal1 = (0, 0, 1, 0.05)
env.add_goal(goal1[0], goal1[1], goal1[2], goal1[3])


# ########### DEFINE POLICY ############################################################################################
policy = policies.SimplePolicy([0, 0, 0, 0, 0.8, 0, 0, 0])



#
N = 1
T = 1000
trajs = utils.collect_trajectories(env, policy, T, N)








test_traj = trajs[0]
moves_x = [t[0] for t in test_traj["states"]]
moves_y = [t[1] for t in test_traj["states"]]
env.plot()
plt.plot(moves_x, moves_y, color="C3", marker="o")