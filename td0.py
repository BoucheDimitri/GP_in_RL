import numpy as np


def temporal_difference0(env, x, r, gamma):
    T = len(x)
    V = np.zeros((env.discrete_rep[1].shape[0], env.discrete_rep[0].shape[0]))
    count_mat = np.ones((env.discrete_rep[1].shape[0], env.discrete_rep[0].shape[0]))
    for t in range(1, T):
        xtminus1 = env.get_discrete_state(x[t - 1])
        print(xtminus1)
        xt = env.get_discrete_state(x[t])
        deltat = r[t - 1] + gamma * V[xt[1], xt[0]] - V[xtminus1[1], xtminus1[0]]
        V[xtminus1[1], xtminus1[0]] += (1 / count_mat[xtminus1[1], xtminus1[0]]) * deltat
        count_mat[xtminus1[1], xtminus1[0]] += 1
    return V

