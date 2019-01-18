import numpy as np

import gptd
import policies
import utils


def gptd_improve(env, Nimprov, N, T, kernel, gamma, nu, sigma0, eps=0.2):
    policy = policies.SimplePolicy([0, 0, 0, 0, 0.8, 0, 0, 0])
    cum_rewards = []
    for n in range(Nimprov):
        trajs = utils.collect_trajectories(env, policy, T, N)
        # Stack trajectories in lists
        xlist = [tr["states"] for tr in trajs]
        rlist = [tr["rewards"] for tr in trajs]
        # Concatenate trajectories
        xconcat, rconcat = utils.concatenate_trajectories(xlist, rlist)
        # Run GPTD for value function estimation
        xdict, alpha, C = gptd.iterate_gptd(xconcat, rconcat, gamma, nu, sigma0, kernel)
        cum_rewards.append(np.sum(rconcat))
        # Improve policy
        policy = policies.GreedyImprovedPolicy(alpha, C, xdict, kernel, eps)
        print(n)
    return cum_rewards
