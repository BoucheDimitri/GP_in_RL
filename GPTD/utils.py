import numpy as np
import copy
import Bayesian_Policy_Gradient_Algo as bpg


class Adam:
    def __init__(self, alpha=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha, self.beta1, self.beta2, self.epsilon = alpha, beta1, beta2, epsilon
    def reset(self):
        self.i, self.n, self.u, self.f, self.hat_n, self.hat_u = 0, 0, 0, 0, 0, 0
    def update(self, grad):
        self.i += 1
        self.f = grad
        self.n = self.beta1 * self.n + (1 - self.beta1) * self.f
        self.u = self.beta2 * self.u + (1 - self.beta2) * self.f ** 2
        self.hat_n = self.n / (1 - self.beta1 ** self.i)
        self.hat_u = self.u / (1 - self.beta2 ** self.i)
        return (self.alpha * self.hat_n) / (np.sqrt(self.hat_u) + self.epsilon)       
        

def collect_episodes(mdp, policy=None, horizon=None, n_episodes=1):
    paths = []

    for _ in range(n_episodes):
        observations = []
        actions = []
        rewards = []
        next_states = []
        state = mdp.reset()
        for _ in range(horizon):
            action = policy.draw_action(state)
            next_state, reward, terminal, _ = mdp.step(action)
            observations.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            state = copy.copy(next_state)
            if terminal:
                # Finish rollout if terminal state reached
                break
                # We need to compute the empirical return for each time step along the
                # trajectory
        paths.append(dict(
            states=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_states=np.array(next_states)
        ))
    return paths


def estimate_performance(mdp, policy=None, horizon=None, n_episodes=1):
    ''' We return the average sum of rewards on one path'''
    paths = collect_episodes(mdp, policy, horizon, n_episodes)
    return np.sum([p["rewards"] for p in paths])/n_episodes


def gradient_MC(policy, paths, theta):
    M = len(paths)
    grad = np.zeros(len(theta))
    for p in paths:
        state, action, reward = p["states"], p["actions"], p["rewards"]
        T = len(reward)
        grad += np.sum([reward[t]*policy.log_gradient(state[t], action[t]) for t in range(T)], axis=0)
    return grad/M
    
    
def Bayesian_Quadrature(mdp, policy, M, N, theta, sparse, sigma, alpha, alphaN, n_episodes = 100):
    BPG1 = bpg.BPG(mdp, policy, theta.copy(), alpha, N, M, horizon=20)
    BPG1.theta_update(model=1, gradient='regular', sparse=sparse, sigma=sigma)
    policies_BPG1 = BPG1.get_policies()

    BPG2 = bpg.BPG(mdp, policy, theta.copy(), alpha, N, M, horizon=20)
    BPG2.theta_update(model=2, gradient='regular', sparse=sparse, sigma=sigma)
    policies_BPG2 = BPG2.get_policies()

    BPGN1 = bpg.BPG(mdp, policy, theta.copy(), alphaN, N, M, horizon=20)
    BPGN1.theta_update(model=1, gradient='natural', sparse=sparse, sigma=sigma)
    policies_BPGN1 = BPGN1.get_policies()

    BPGN2 = bpg.BPG(mdp, policy, theta.copy(), alphaN, N, M, horizon=20)
    BPGN2.theta_update(model=2, gradient='natural', sparse=sparse, sigma=sigma)
    policies_BPGN2 = BPGN2.get_policies()
    
    # Mean Squared Error
    perf_BPG1 = [estimate_performance(mdp, policy=pi, horizon=20, n_episodes=n_episodes) for pi in policies_BPG1]
    perf_BPG2 = [estimate_performance(mdp, policy=pi, horizon=20, n_episodes=n_episodes) for pi in policies_BPG2]
    perf_BPGN1  = [estimate_performance(mdp, policy=pi, horizon=20, n_episodes=n_episodes) for pi in policies_BPGN1]
    perf_BPGN2 = [estimate_performance(mdp, policy=pi, horizon=20, n_episodes=n_episodes) for pi in policies_BPGN2]
    return perf_BPG1, perf_BPG2, perf_BPGN1, perf_BPGN2


def BPG_MC(mdp, policy, M, N, theta_initial, n_episodes, gamma, epochs=1, T=20):
    list_avg_returns = []
    for e in range(epochs):
        theta = theta_initial.copy() # we initialize theta
        avg_return = []
        for i in range(N):
            pi = policy(*list(theta))
            paths = collect_episodes(mdp, policy=pi, horizon=T, n_episodes=M)
            grad = gradient_MC(pi, paths, theta)
            theta = theta.copy() + gamma[i]*grad   
            avg = estimate_performance(mdp, policy=pi, horizon=T, n_episodes=n_episodes)
            avg_return.append(avg)
        pi = policy(*list(theta))
        avg = estimate_performance(mdp, policy=pi, horizon=T, n_episodes=n_episodes)
        avg_return.append(avg)
        list_avg_returns.append(avg_return)
    list_avg_returns = np.array(list_avg_returns)
    avg_returnsMC = np.mean(list_avg_returns, axis=0)
    return avg_returnsMC
