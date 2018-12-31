import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Maze:

    def __init__(self, len_x, len_y, pace, noise=0.05, rebound=0.01):
        self.len_x = len_x
        self.len_y = len_y
        self.barriers = []
        self.goals = []
        self.pace = pace
        self.noise = noise
        self.rebound = rebound
        self.actions_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        self.opposite_actions = [4, 5, 6, 7, 0, 1, 2, 3]

    def add_barrier(self, start, stop, a, b):
        self.barriers.append((start, stop, a, b))

    def add_goal(self, origin_x, origin_y, len_x, len_y):
        self.goals.append((origin_x, origin_y, len_x, len_y))

    def get_boundaries(self):
        return np.zeros(2), np.array([self.len_x, self.len_y])

    # @staticmethod
    # def barrier_test_diag_horizontal(coords_init, coords_moved, barrier):
    #     c = (coords_moved[1] - coords_init[1]) / (coords_moved[0] - coords_init[0])
    #     d = coords_init[1] - c * coords_init[0]
    #     # Paralell move so no crossing
    #     if c == barrier[2]:
    #         cross = False
    #         return cross
    #     crossx = (barrier[3] - d) / (c - barrier[2])
    #     top_coord = max(coords_init[0], coords_moved[0])
    #     bot_coord = min(coords_init[0], coords_moved[0])
    #     cross = (barrier[0] <= crossx <= barrier[1]) and (bot_coord <= crossx <= top_coord)
    #     return cross


    @staticmethod
    def barrier_step_diag_horizontal(coords_init, coords_moved, action, barrier, rebound):
        c = (coords_moved[1] - coords_init[1]) / (coords_moved[0] - coords_init[0])
        d = coords_init[1] - c * coords_init[0]
        # Paralell move so no crossing
        if c == barrier[2]:
            return coords_moved
        crossx = (barrier[3] - d) / (c - barrier[2])
        top_coord = max(coords_init[0], coords_moved[0])
        bot_coord = min(coords_init[0], coords_moved[0])
        cross = (barrier[0] <= crossx <= barrier[1]) and (bot_coord <= crossx <= top_coord)
        if cross:
            coords_moved = crossx, barrier[2] * crossx + barrier[3]
            coords_moved = Maze.move_coords(coords_moved, action, - rebound)
            return coords_moved
        else:
            return coords_moved

    # @staticmethod
    # def barrier_test_vertical(coords_init, coords_moved, barrier):
    #     if coords_init[0] < barrier[0] or coords_init[0] > barrier[1]:
    #         cross = False
    #         return cross
    #     else:
    #         y = barrier[2] * coords_init[0] + barrier[3]
    #         top_coord = max(coords_init[1], coords_moved[1])
    #         bot_coord = min(coords_init[1], coords_moved[1])
    #         if bot_coord <= y <= top_coord:
    #             cross = True
    #         else:
    #             cross = False
    #         return cross

    @staticmethod
    def barrier_step_vertical(coords_init, coords_moved, action, barrier, rebound):
        if coords_init[0] < barrier[0] or coords_init[0] > barrier[1]:
            return coords_moved
        else:
            y = barrier[2] * coords_init[0] + barrier[3]
            top_coord = max(coords_init[1], coords_moved[1])
            bot_coord = min(coords_init[1], coords_moved[1])
            if bot_coord <= y <= top_coord:
                coords_moved = coords_init[0], y
                coords_moved = Maze.move_coords(coords_moved, action, - rebound)
                return coords_moved
            else:
                return coords_moved

    # @staticmethod
    # def barrier_test(coords_init, coords_moved, barrier):
    #     if coords_init[0] == coords_moved[0]:
    #         return Maze.barrier_test_vertical(coords_init, coords_moved, barrier)
    #     else:
    #         return Maze.barrier_test_diag_horizontal(coords_init, coords_moved, barrier)

    @staticmethod
    def barrier_step(coords_init, coords_moved, action, barrier, rebound):
        if coords_init[0] == coords_moved[0]:
            return Maze.barrier_step_vertical(coords_init, coords_moved, action, barrier, rebound)
        else:
            return Maze.barrier_step_diag_horizontal(coords_init, coords_moved, action, barrier, rebound)

    @staticmethod
    def move_coords(coords, action, pace):
        if action == 0:
            return coords[0], coords[1] + pace
        elif action == 1:
            sqrt2 = np.sqrt(2)
            return coords[0] + pace / sqrt2, coords[1] + pace / sqrt2
        elif action == 2:
            return coords[0] + pace, coords[1]
        elif action == 3:
            sqrt2 = np.sqrt(2)
            return coords[0] + pace / sqrt2, coords[1] - pace / sqrt2
        elif action == 4:
            return coords[0], coords[1] - pace
        elif action == 5:
            sqrt2 = np.sqrt(2)
            return coords[0] - pace / sqrt2, coords[1] - pace / sqrt2
        elif action == 6:
            return coords[0] - pace, coords[1]
        else:
            sqrt2 = np.sqrt(2)
            return coords[0] - pace / sqrt2, coords[1] + pace / sqrt2

    def is_terminal(self, state):
        for goal in self.goals:
            if goal[0] <= state[0] <= goal[0] + goal[2] and goal[1] <= state[1] <= goal[1] + goal[3]:
                return True
        return False

    # def is_available(self, coords, action):
    #     coords_moved = Maze.move_coords(coords, action, self.pace)
    #     cross = Maze.barrier_test(coords, coords_moved, self.barriers[0])
    #     return not cross
    #
    # def available_actions(self, coords):
    #     possible_actions = []
    #     for action in self.actions_names:
    #         possible = True
    #         coords_moved = Maze.move_coords(coords, action, self.pace)
    #         for barrier in self.barriers:
    #             cross = Maze.barrier_test(coords, coords_moved, barrier)
    #             if cross:
    #                 possible = False
    #                 break
    #         if possible:
    #             possible_actions.append(action)
    #     return possible_actions

    def step(self, state, action):
        # success = np.random.binomial(1, 1 - self.noise)
        # if success == 0:
        #     action = self.opposite_actions[action]
        noise = np.random.normal(0, self.noise)
        next_state = Maze.move_coords(state, action, self.pace + noise)
        amin, amax = self.get_boundaries()
        next_state = np.clip(next_state, amin, amax)
        for barrier in self.barriers:
            next_state = Maze.barrier_step(state, next_state, action, barrier, self.rebound)
        isterminal = self.is_terminal(next_state)
        if not isterminal:
            reward = -1
        else:
            reward = 0
        return next_state, reward, isterminal

    def reset_gaussian(self, mu, sig):
        init = np.random.multivariate_normal(mu, sig)
        amin, amax = self.get_boundaries()
        return np.clip(init, amin, amax)

    def reset(self):
        x = np.random.uniform(0, self.len_x)
        y = np.random.uniform(0, self.len_y)
        return x, y

    @staticmethod
    def plot_barrier(barrier, ax, npoints):
        linspace = np.linspace(barrier[0], barrier[1], npoints)
        ax.plot(linspace, [barrier[2] * t + barrier[3] for t in linspace], color="C0")

    @staticmethod
    def plot_goal(goal, ax):
        ax.add_patch(patches.Rectangle((goal[0], goal[1]), goal[2], goal[3], color="C1"))

    def plot(self, npoints=50):
        fig, ax = plt.subplots()
        for barrier in self.barriers:
            Maze.plot_barrier(barrier, ax, npoints)
        for goal in self.goals:
            Maze.plot_goal(goal, ax)
        ax.set_xlim(0, self.len_x)
        ax.set_ylim(0, self.len_y)









