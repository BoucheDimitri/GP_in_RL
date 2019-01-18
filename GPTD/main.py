import numpy as np
import importlib
import matplotlib.pyplot as plt
import os
import pickle

import maze
import utils
import policies
import offline_benchmark as off_bench
import gptd
import td
import gptd_episodic_improv as improv

# Plot parameters
plt.rcParams.update({"font.size": 35})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})

# Reload modules, for developpement
importlib.reload(maze)
importlib.reload(utils)
importlib.reload(off_bench)
importlib.reload(gptd)
importlib.reload(td)
importlib.reload(improv)


# ############# BUILD MAZE #############################################################################################
env = maze.Maze(1, 1, 0.1, 0.03, 0.01)

# Set barriers
barrier1 = (-0.1, 0.4, -0.75, 0.8)
barrier2 = (0.55, 0.7, -3, 2.525)
barrier3 = (0.7, 1.1, -0.5, 0.6)
barrier4 = (0.2, 0.55, 0.5, 0.1)
env.add_barrier(barrier1[0], barrier1[1], barrier1[2], barrier1[3])
env.add_barrier(barrier2[0], barrier2[1], barrier2[2], barrier2[3])
env.add_barrier(barrier3[0], barrier3[1], barrier3[2], barrier3[3])
env.add_barrier(barrier4[0], barrier4[1], barrier4[2], barrier4[3])

# Set goal regions
goal1 = (0, 0, 1, 0.05)
env.add_goal(goal1[0], goal1[1], goal1[2], goal1[3])


# ########### SIMULATE TRAJECTORIES ####################################################################################
# Define policy
policy = policies.SimplePolicy([0, 0, 0, 0, 0.8, 0, 0, 0])

# Simulate trajectories
N = 10
T = 1000
trajs = utils.collect_trajectories(env, policy, T, N)

# Stack trajectories in lists
xlist = [tr["states"] for tr in trajs]
rlist = [tr["rewards"] for tr in trajs]

# Concatenate trajectories
xconcat, rconcat = utils.concatenate_trajectories(xlist, rlist)


# ########## GPTD ######################################################################################################
# Initialize Kernel
kernel = gptd.Gaussian_Kernel(sigma=0.1)

# Parameters
gamma = 0.95
nu = 0.05
sigma0 = 1

# Rune GPTD
xdict, alpha, C = gptd.iterate_gptd(xconcat, rconcat, gamma, nu, sigma0, kernel)

# Vizualization
coords1 = np.linspace(0, 1, 100)
coords2 = np.linspace(0, 1, 100)
M, S = gptd.mean_variance_matrices(xdict, coords1, coords2, alpha, C, kernel)
moves_x, moves_y = utils.trajectory_to_moves(xconcat)
ax1 = utils.visualization_2D_discrete(env, coords1, coords2, S, moves=None)
# ax1.set_title("GPTD - Estimated Variance Map")
ax2 = utils.visualization_2D_discrete(env, coords1, coords2, M, moves=(moves_x, moves_y))
# ax2.set_title("GPTD - Estimated Value Map")


# ########## TD0 #######################################################################################################
# Discretize state space
env.build_discretization(50, 50)

V = td.temporal_difference0(env, xconcat, rconcat, gamma)

ax3 = utils.visualization_2D_discrete(env, env.discrete_rep[0], env.discrete_rep[1], V, moves=None)
# ax3.set_title("TD0 - Estimated Value Map")



# ######## MC ESTIMATION ###############################################################################################
# Load value map objective from pickle object
with open(os.path.join(os.getcwd(), "VTD0.pickle"), "rb") as inp:
    Vob = pickle.load(inp)

Ngrid = np.arange(1, 20, 1)
Navg = 100
errors_gptd, errors_td0 = utils.comp_gptd_td0(env, Vob, policy, Ngrid, Navg, T, gamma, nu, sigma0, kernel)

# with open(os.path.join(os.getcwd(), "errors_GPTD.pickle"), "wb") as outp:
#     pickle.dump(errors_gptd, outp, pickle.HIGHEST_PROTOCOL)
#
#
# with open(os.path.join(os.getcwd(), "errors_TD0.pickle"), "wb") as outp:
#     pickle.dump(errors_td0, outp, pickle.HIGHEST_PROTOCOL)
#
#
# with open(os.path.join(os.getcwd(), "errors_GPTD.pickle"), "rb") as inp:
#     errors_gptd = pickle.load(inp)
#
#
# with open(os.path.join(os.getcwd(), "errors_TD0.pickle"), "rb") as inp:
#     errors_td0 = pickle.load(inp)
#






# ################" OFFLINE BENCHMARK ###############################################################""
#
# xdict_bis = utils.trajectory_list_to_ndarray(x)
# rbis = np.array(r)
#
# t = xdict_bis.shape[1]
# H = off_bench.form_H(t, gamma)
# Q = off_bench.form_Q(xdict_bis, H, sigma0, kernel)
# alpha = off_bench.compute_alpha(H, Q, rbis[:t-1])
# C = off_bench.compute_C(H, Q)
#
#
# coord1, coord2 = np.meshgrid(np.arange(0, 1.05, 0.05), np.arange(0, 1.05, 0.05))
# nx = coord1.shape[0]
# M = np.zeros((coord1.shape[0], coord1.shape[0]))
# S = np.zeros((coord1.shape[0], coord1.shape[0]))
#
# for i in range(0, nx):
#     for j in range(0, nx):
#         xnew = np.array([coord1[i, j], coord2[i, j]])
#         v, p = off_bench.get_mean_variance(xdict_bis, xnew, alpha, C, kernel)
#         M[i, j] = v
#         S[i, j] = p
#
#
#
# plt.figure()
# env.plot()
# plt.contourf(coord1, coord2, M)
# plt.plot(moves_x, moves_y, color="C3", marker="o")
# plt.colorbar()
#
#
#
#


































