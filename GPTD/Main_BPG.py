import numpy as np
import matplotlib.pyplot as plt
import copy
import Policies as pol
from Bayesian_Policy_Gradient_Algo import *

plt.ion()
plt.show()


## Experimentation 1: 
# We initialized the parameters:
mdp = pol.Linear_Quadratic_Regulator()
policy = pol.policy_Linear_Quadratic_Regulator

M = 5
N = 60
n_episodes = 150
theta_initial = copy.copy(np.array([-.2, 1.]))
sparse = False
sigma = 0.05*np.sqrt(20)

beta0 = -0.001/(20+np.arange(N))
betaN0 = -0.02*np.ones(N)/(20+np.arange(N))
beta1 = -0.006/(20+np.arange(N))
betaN1 = -0.1*np.ones(N)/(20+np.arange(N))
alpha = np.vstack((beta0,beta1)).T
alphaN = np.vstack((betaN0,betaN1)).T

gamma0 = -0.01/(20+np.arange(N))
gamma1 = -0.05/(20+np.arange(N))
gamma = np.vstack((gamma0,gamma1)).T

# We calcul the Average Return with one epoch:
performances_BPG1, performances_BPG2, performances_BPGN1, performances_BPGN2 =  Bayesian_Quadrature(mdp, policy, M, N, theta_initial,sparse, sigma, alpha, alphaN, n_episodes)
avg_returnsMC = utl.BPG_MC(mdp, policy, M, N, theta_initial, n_episodes, gamma, epochs=1, T=20)

# We display the figures:
plt.figure(1)
plt.clf()
plt.plot(avg_returnsMC, label='MC')
plt.plot(performances_BPG1, label='performances_BPG1')
plt.plot(performances_BPG2, label='performances_BPG2')
plt.plot(performances_BPGN1, label='performances_BPGN1')
plt.plot(performances_BPGN2, label='performances_BPGN2')
plt.ylabel('Average Expected Return')
plt.xlabel('Number of Updates (Sample Size = %s)' %(M))
plt.title('Average returns')
plt.legend()

## Experimentation 2: Random theta
# We initialized the parameters:
mdp = pol.Linear_Quadratic_Regulator()
policy = pol.policy_Linear_Quadratic_Regulator_modified

M = 5
N = 60
Lambda = np.random.normal(1,1)
sigma = np.random.rand()+0.1
theta_initial = copy.copy(np.array([Lambda,sigma]))
sparse = False
sigma = 0.01*np.sqrt(20)

beta0 = -20/(20+np.arange(N))*0.02
betaN0 = -20*np.ones(N)/(20+np.arange(N))*0.06
beta1 = -20/(20+np.arange(N))*0.1
betaN1 = -20*np.ones(N)/(20+np.arange(N))
alpha = np.vstack((beta0,beta1)).T
alphaN = np.vstack((betaN0,betaN1)).T

# We calcul the Average Return with one epoch:
performances_BPG1,performances_BPG2,performances_BPGN1,performances_BPGN2 =  Bayesian_Quadrature(mdp, policy, M, N, theta_initial,sparse, sigma, alpha, alphaN, n_episodes)
avg_returnsMC = utl.BPG_MC(mdp, policy, M, N, theta_initial, n_episodes, gamma, epochs=1, T=20)

# We display the figures:
plt.figure(2)
plt.clf()
plt.plot(avg_returnsMC, label='MC')
plt.plot(performances_BPG1, label='performances_BPG1')
plt.plot(performances_BPG2, label='performances_BPG2')
plt.plot(performances_BPGN1, label='performances_BPGN1')
plt.plot(performances_BPGN2, label='performances_BPGN2')
plt.ylabel('Average Expected Return')
plt.xlabel('Number of Updates (Sample Size = %s)' %(M))
plt.title('Average returns')
plt.legend()

## Experimentation 3: Sparse
# We initialized the parameters:
mdp = pol.Linear_Quadratic_Regulator()
policy = pol.policy_Linear_Quadratic_Regulator_modified

M = 20
N = 60
theta_initial = copy.copy(np.array([1., 0.1]))
sparse = True
sigma = 0.1*np.sqrt(20)

beta0 = -20/(20+np.arange(N))*0.02
betaN0 = -20*np.ones(N)/(20+np.arange(N))*0.06
beta1 = -20/(20+np.arange(N))*0.1
betaN1 = -20*np.ones(N)/(20+np.arange(N))
alpha = np.vstack((beta0,beta1)).T
alphaN = np.vstack((betaN0,betaN1)).T

# We calcul the Average Return with one epoch:
performances_BPG1,performances_BPG2,performances_BPGN1,performances_BPGN2 =  Bayesian_Quadrature(mdp, policy, M, N, theta_initial, sparse, sigma, alpha, alphaN, n_episodes)
avg_returnsMC = utl.BPG_MC(mdp, policy, M, N, theta_initial, n_episodes, gamma, epochs=1, T=20)

# We display the figures:
plt.figure(3)
plt.clf()
plt.plot(avg_returnsMC, label='MC')
plt.plot(performances_BPG1, label='performances_BPG1')
plt.plot(performances_BPG2, label='performances_BPG2')
plt.plot(performances_BPGN1, label='performances_BPGN1')
plt.plot(performances_BPGN2, label='performances_BPGN2')
plt.ylabel('Average Expected Return')
plt.xlabel('Number of Updates (Sample Size = %s)' %(M))
plt.title('Average returns')
plt.legend()
