import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import gptd
import td


def collect_trajectories(env, policy, T, N):
    trajs = []
    for n in range(N):
        terminal = False
        t = 0
        s = env.reset()
        traj = {"states": [s], "actions": [], "rewards": []}
        while t < T and not terminal:
            a = policy.draw_action(s)
            next_s, r, terminal = env.step(s, a)
            s = next_s
            traj["states"].append(s)
            traj["actions"].append(a)
            traj["rewards"].append(r)
            t += 1
        traj["length"] = t
        trajs.append(traj)
    return trajs


def trajectory_list_to_ndarray(xtrajs):
    ntrajs = len(xtrajs)
    xarray = np.zeros((2, ntrajs))
    for i in range(ntrajs):
        xarray[:, i] = xtrajs[i]
    return xarray


def concatenate_trajectories(xs, rs):
    xconcat = []
    rconcat = []
    for x in xs:
        xconcat += x
    for r in rs:
        rconcat += r + [-1]
    return xconcat, rconcat[:len(rconcat) - 1]


def trajectory_to_moves(x):
    moves_x = [t[0] for t in x]
    moves_y = [t[1] for t in x]
    return moves_x, moves_y


def visualization_2D(env, coords1, coords2, M, moves):
    fig, ax = plt.subplots()
    env.plot(ax)
    if moves:
        ax.scatter(moves[0], moves[1], c="C3")
    mappable = ax.contourf(coords1, coords2, M, cmap=cm.coolwarm)
    plt.colorbar(mappable, ax=ax)
    return ax


def visualization_2D_discrete(env, coords1, coords2, M, moves):
    fig, ax = plt.subplots()
    env.plot(ax)
    if moves:
        ax.scatter(moves[0], moves[1], c="C3")
    mappable = ax.pcolor(coords1, coords2, M, cmap=cm.coolwarm)
    plt.colorbar(mappable, ax=ax)
    return ax


def comp_gptd_td0(env, Vob, policy, Ngrid, Navg, T, gamma, nu, sigma0, kernel):
    error_gptd = []
    error_td0 = []
    for N in Ngrid:
        egptd = 0
        etd0 = 0
        for i in range(Navg):
            trajs = collect_trajectories(env, policy, T, N)
            # Stack trajectories in lists
            xlist = [tr["states"] for tr in trajs]
            rlist = [tr["rewards"] for tr in trajs]
            # Concatenate trajectories
            xconcat, rconcat = concatenate_trajectories(xlist, rlist)
            # Rune GPTD
            xdict, alpha, C = gptd.iterate_gptd(xconcat, rconcat, gamma, nu, sigma0, kernel)
            # Discretization (take the middle of the squares
            coords1 = env.discrete_rep[0] + (1 / (2 * env.discrete_rep[0].shape[0]))
            coords2 = env.discrete_rep[1] + (1 / (2 * env.discrete_rep[1].shape[0]))
            Vgptd, S = gptd.mean_variance_matrices(xdict, coords1, coords2, alpha, C, kernel)
            Vtd0 = td.temporal_difference0(env, xconcat, rconcat, gamma)
            egptd += (1 / Navg) * np.mean((Vgptd - Vob) ** 2)
            etd0 += (1 / Navg) * np.mean((Vtd0 - Vob) ** 2)
        error_gptd.append(egptd)
        error_td0.append(etd0)
        print(N)
    return error_gptd, error_td0







































