import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
        ax.plot(moves[0], moves[1], color="C3", marker="o")
    mappable = ax.contourf(coords1, coords2, M, cmap=cm.coolwarm)
    plt.colorbar(mappable, ax=ax)







































