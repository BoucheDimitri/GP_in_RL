import numpy as np
import bisect


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
