import numpy as np


class policy_Linear_Quadratic_Regulator:
    def __init__(self, Lambda=-.2, sigma=1):
        self.Lambda = Lambda
        self.sigma = sigma
    def draw_action(self, state):
        action = np.random.normal(self.Lambda*state, self.sigma**2)
        return action
    def log_gradient(self, state, action):
        x1 = (action-self.Lambda*state)/self.sigma**2
        x2 = (action-self.Lambda*state)**2/self.sigma**3-1/self.sigma
        return np.array([x1,x2])


class policy_Linear_Quadratic_Regulator_modified:
    ''' We randomly initialize, in order to avoid diverging, we define Lambda and Sigma by a change of variable '''
    def __init__(self, p1=0, p2=0):
        self.p1 = p1
        self.p2 = p2
    def draw_action(self, state):
        self.Lambda, self.sigma = -1.999 + 1.998/(1+np.exp(self.p1)), .001 + 1/(1+np.exp(self.p2))
        action = np.random.normal(self.Lambda*state, self.sigma**2)
        return action
    def log_gradient(self, state, action):
        x1 = (self.Lambda*state-action)*np.exp(self.p1)/((1+np.exp(self.p1))*self.sigma)**2
        x2 = ((action-self.Lambda*state)**2/self.sigma**3-1/self.sigma)*np.exp(self.p2)/(1+np.exp(self.p2))**2
        return np.array([x1,x2])


class Linear_Quadratic_Regulator:
    def __init__(self):
        self.reset()
    def step(self, action):
        reward = self.state**2 + .1*action**2 + np.random.normal(0,.1)
        self.state = self.state + action + np.random.normal(0,.01)
        return self.state, reward, False, {}
    def reset(self):
        self.state = np.random.normal(.3,.001)
        return self.state