import numpy as np
import scipy.stats as stats


class Gaussian_Kernel:

    def __init__(self, sigma):
        self.sigma = sigma

    def k(self, x1, x2):
        return stats.multivariate_normal(0, self.sigma).pdf(x1 - x2)


def compute_k(xdict, xnew, kernel):
    n = xdict.shape[1]
    k = np.array([kernel.k(xdict[:, i], xnew) for i in range(n)])
    return k


def compute_a(k, Kinv):
    return np.dot(Kinv, k)


def compute_epsilon(ktt, k, a):
    return ktt - np.dot(k.T, a)


def augment_Kinv(epsilon, a, Kinv):
    nminus1 = Kinv.shape[0]
    acol = a.reshape((a.shape[0], 1))
    aaT = np.dot(acol, acol.T)
    Kinv_agmented = np.zeros((nminus1 + 1, nminus1 + 1))
    firstblock = epsilon * Kinv + aaT
    Kinv_agmented[0:nminus1, 0:nminus1] = firstblock
    Kinv_agmented[-1, -1] = 1
    Kinv_agmented[-1, 0:nminus1] = acol.T
    Kinv_agmented[0:nminus1, -1] = acol
    return (1 / epsilon) * Kinv_agmented




kernel = Gaussian_Kernel(1)

xdict = np.array([[2, 3, 4], [0, 1, 2]])
xnew = np.array([1, 1])

test = compute_k(xdict, xnew, kernel)