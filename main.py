import numpy as np
import importlib
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

import maze
import utils
import policies
import gptd
import offline_benchmark as off_bench

importlib.reload(maze)
importlib.reload(utils)
importlib.reload(gptd)
importlib.reload(off_bench)


# ############# BUILD MAZE #############################################################################################
env = maze.Maze(1, 1, 0.1, 0.03, 0.01)

barrier1 = (-0.1, 0.4, -0.75, 0.8)
barrier2 = (0.55, 0.7, -3, 2.525)
barrier3 = (0.7, 1.1, -0.5, 0.6)
barrier4 = (0.2, 0.55, 0.5, 0.1)

env.add_barrier(barrier1[0], barrier1[1], barrier1[2], barrier1[3])
env.add_barrier(barrier2[0], barrier2[1], barrier2[2], barrier2[3])
env.add_barrier(barrier3[0], barrier3[1], barrier3[2], barrier3[3])
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

xconcat = []
rconcat = []
for n in range(N):
    xconcat += trajs[n]["states"]
    r = trajs[n]["rewards"]
    r.reverse()
    r.append(-1)
    r.reverse()
    rconcat += r


# Plot maze and trajectory
test_traj = trajs[0]
moves_x = [t[0] for t in x]
moves_y = [t[1] for t in x]
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
at = gptd.compute_a(kt, Kinv)
eps = gptd.compute_epsilon(ktt, kt, at)
# Add to dictionnary test
if eps > nu:
    xdict = np.concatenate((xdict, x[1].reshape((2, 1))), axis=1)
    Kinv = gptd.augment_Kinv(eps, at, Kinv)
    # We initialize H instead of updating it for the first augmentation step
    H = gptd.initialize_H(gamma)
    # We intialize Q instead of updating it for the first augmentation step
    # For that we need to create the kernel matrix
    K = gptd.compute_K(xdict, kernel)
    Q = gptd.intialize_Q(K, H, sigma0)
    # At first step we compute alpha directly since we cannot define the recursions
    alpha = gptd.initialize_alpha(H, Q, np.array([-1]))
    # At first step we compute C directly since we cannot define the recursions
    C = gptd.initialize_C(H, Q)


# Second step (t=3)

xdict, Kinv, H, Q, ahat_t,at, alpha, C = gptd.first_step(x[0], x[1], -1, gamma, sigma0, kernel)

atminus1 = at
ahat_tminus1 = ahat_t

test, ahat_t, k, eps = gptd.test_phase(xdict, x[2], Kinv, nu, kernel)

#
# deltak_tminus1 = gptd.compute_k(xdict, xdict[:, -1], kernel) - gamma * k
#
# deltak_tt = gptd.compute_deltak_tt(atminus1, deltak_tminus1, k, kernel.k(x[2], x[2]), gamma)

# Kinv = gptd.augment_Kinv(eps, ahat_t, Kinv)
# Ht = gptd.augment_H_case2(H, atminus1, gamma)
# deltak_tminus1 = gptd.compute_k(xdict, xdict[:, -1], kernel) - gamma * k
# gt = gptd.compute_g(Q, H, deltak_tminus1)
# st = gptd.compute_hat_s(atminus1, kt, kernel.k(x[2], x[2]), sigma0, deltak_tminus1, C, gamma)
# ct = gptd.compute_hat_c(H, gt, atminus1)
# alpha_t = gptd.update_alpha_case2(alpha, deltak_tminus1, ct, st, -1, gamma)
# Ct = gptd.update_C_case2(C, ct, st, gamma)
# Qt = gptd.augment_Q(Q, st, gt)
# xdict = np.concatenate((xdict, x[2].reshape((2, 1))), axis=1)


if not test:
    # atminus1 = ahat_tminus1
    H, Q, alpha, C = gptd.iterate_case1(xdict, r[1], atminus1, ahat_t, H, Q, k, alpha, C, gamma, sigma0, kernel)
else:
    # atminus1 =
    xdict, Kinv, H, Q, alpha, C = gptd.iterate_case2(xdict, x[2], r[1], atminus1, ahat_t, eps, Kinv, H, Q, k, alpha, C, gamma, sigma0, kernel)
    atminus1 = np.zeros(xdict.shape[1])
    atminus1[-1] = 1



alpha, C, xdict, Kinv, H, Q = gptd.iterate_gptd(x, r, gamma, nu, sigma0, kernel)


alpha = H.T.dot(Q).dot(r)
C = H.T.dot(Q).dot(H)


coord1, coord2 = np.meshgrid(np.arange(0, 1.05, 0.05), np.arange(0, 1.05, 0.05))
# M = np.zeros((coord1.shape[0], coord1.shape[0]))
S = np.zeros((coord1.shape[0], coord1.shape[0]))
nx = coord1.shape[0]

for i in range(0, nx):
    for j in range(0, nx):
        xnew = np.array([coord1[i, j], coord2[i, j]])
        v, p = gptd.compute_state_mean_variance(xdict, xnew, alpha, C, kernel)
        # M[i, j] = v
        S[i, j] = p



plt.figure()
env.plot()
plt.plot(moves_x, moves_y, color="C3", marker="o")
plt.contourf(coord1, coord2, S)
plt.colorbar()






################" OFFLINE BENCHMARK ###############################################################""

xdict_bis = utils.trajectory_list_to_ndarray(x)
rbis = np.array(r)

t = xdict_bis.shape[1]
H = off_bench.form_H(t, gamma)
Q = off_bench.form_Q(xdict_bis, H, sigma0, kernel)
alpha = off_bench.compute_alpha(H, Q, rbis[:t-1])
C = off_bench.compute_C(H, Q)


coord1, coord2 = np.meshgrid(np.arange(0, 1.05, 0.05), np.arange(0, 1.05, 0.05))
nx = coord1.shape[0]
M = np.zeros((coord1.shape[0], coord1.shape[0]))
S = np.zeros((coord1.shape[0], coord1.shape[0]))

for i in range(0, nx):
    for j in range(0, nx):
        xnew = np.array([coord1[i, j], coord2[i, j]])
        v, p = off_bench.get_mean_variance(xdict_bis, xnew, alpha, C, kernel)
        M[i, j] = v
        S[i, j] = p



plt.figure()
env.plot()
plt.contourf(coord1, coord2, M)
plt.plot(moves_x, moves_y, color="C3", marker="o")
plt.colorbar()


xcenter = np.array([0.5, 0.5])
Ktest = np.zeros((coord1.shape[0], coord1.shape[0]))
for i in range(0, n):
    for j in range(0, n):
        xnew = np.array([coord1[i, j], coord2[i, j]])
        print(xnew)
        Ktest[i, j] = kernel.k(xcenter, xnew)



test2 = coord1**2 + coord2**2
plt.figure()
# env.plot()
plt.pcolormesh(coord1, coord2, Ktest)
plt.colorbar()




xfixed = 0.5
test = [kernel.k(np.array([xfixed, y]), xcenter) for y in np.arange(0, 1, 0.01)]











































