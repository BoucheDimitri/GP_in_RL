import numpy as np
import bisect
import gptd
import maze


class SimplePolicy:

    def __init__(self, probas):
        self.probas = probas

    def draw_action(self, state):
        udraw = np.random.uniform(0, 1)
        cum = np.cumsum(self.probas)
        spot = bisect.bisect(cum, udraw)
        # Explore
        if spot == 8:
            spot = np.random.randint(0, 7)
        return spot


class GreedyImprovedPolicy:

    def __init__(self, alpha, C, xdict, kernel, eps, pace=0.1):
        self.alpha = alpha
        self.C = C
        self.eps = eps
        self.pace = pace
        self.xdict = xdict
        self.kernel = kernel

    def draw_action(self, state):
        udraw = np.random.uniform(0, 1)
        # Explore
        if udraw > self.eps:
            return np.random.randint(0, 7)
        # Greedy with respect to next values
        else:
            next_states = [maze.Maze.move_coords(state, a, self.pace) for a in range(0, 8)]
            values = [gptd.compute_state_mean_variance(self.xdict, ns, self.alpha, self.C, self.kernel)[0] for ns in next_states]
            return np.argmax(values)

