import numpy as np


def temporal_difference0(env, x, r, gamma):
    T = len(x)
    V = np.zeros((env.discrete_rep[1].shape[0], env.discrete_rep[0].shape[0]))
    count_mat = np.ones((env.discrete_rep[1].shape[0], env.discrete_rep[0].shape[0]))
    for t in range(1, T):
        xtminus1 = env.get_discrete_state(x[t - 1])
        xt = env.get_discrete_state(x[t])
        deltat = r[t - 1] + gamma * V[xt[1], xt[0]] - V[xtminus1[1], xtminus1[0]]
        V[xtminus1[1], xtminus1[0]] += (1 / count_mat[xtminus1[1], xtminus1[0]]) * deltat
        count_mat[xtminus1[1], xtminus1[0]] += 1
    return V


def temporal_difference1(env, trajs, gamma):
    V = np.zeros((env.discrete_rep[1].shape[0], env.discrete_rep[0].shape[0]))
    count_mat = np.ones((env.discrete_rep[1].shape[0], env.discrete_rep[0].shape[0]))
    for traj in trajs:
        x = traj["states"]
        x0 = env.get_discrete_state(x[0])
        T = len(x)
        r = traj["rewards"]
        R = np.sum(np.array(r) * np.exp(np.array([i * np.log(gamma) for i in range(T - 1)])))
        nvisit = count_mat[x0[1], x0[0]]
        V[x0[1], x0[0]] = ((nvisit - 1) / nvisit) * V[x0[1], x0[0]] + (1 / nvisit) * R
    return V
