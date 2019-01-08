import numpy as np


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


