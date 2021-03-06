import numpy as np


class Gaussian_Kernel:
    """
    Class for Gaussian Kernel
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def k(self, x1, x2):
        """
        Apply the kernel to 2 vectors
        """
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.sigma ** 2))


def compute_k(xdict, xnew, kernel):
    """
    Compute the vector which entries are k(x, xnew) for x in xdict

    Params:
        xdict (np.ndarray): Representative states stacked in columns in an array
        xnew (np.ndarrya): New state
        kernel (Gaussian_Kernel): the Gaussian kernel

    Returns:
        np.ndarray: the kernel vector of xnew "against" states in the dict
    """
    n = xdict.shape[1]
    k = np.array([kernel.k(xdict[:, i], xnew) for i in range(n)])
    return k


def compute_K(xdict, kernel):
    """
    Compute the pairwise kernel matrix of xdict

    Params:
        xdict (np.ndarray): Representative states stacked in columns in an array
        kernel (Gaussian_Kernel): the Gaussian kernel
    Returns:
        np.ndarray: Kernel matrix
    """
    n = xdict.shape[1]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = kernel.k(xdict[:, i], xdict[:, j])
            K[j, i] = K[i, j]
    return K


def augment_Kinv(epsilon, a, Kinv):
    """
    Sequential block inversion (inverse is not calculated at each step but augmented to get the next inverse)
    """
    nminus1 = Kinv.shape[0]
    acol = a.reshape((a.shape[0], 1))
    aaT = np.dot(acol, acol.T)
    Kinv_augmented = np.zeros((nminus1 + 1, nminus1 + 1))
    firstblock = epsilon * Kinv + aaT
    Kinv_augmented[0:nminus1, 0:nminus1] = firstblock
    Kinv_augmented[-1, -1] = 1
    Kinv_augmented[-1, 0:nminus1] = - a.T
    Kinv_augmented[0:nminus1, -1] = - a
    return (1 / epsilon) * Kinv_augmented


def augment_C(C):
    """
    Sequential augmentation formula for C
    """
    nr = C.shape[0]
    nc = C.shape[1]
    C = np.concatenate((C, np.zeros((1, nc))), axis=0)
    C = np.concatenate((C, np.zeros((nr + 1, 1))), axis=1)
    return C


def iterate_gptd(x, r, gamma, nu, sigma0, kernel):
    """
    Run sparsfified-online GPTD
    """
    # Intialization
    xdict = x[0].reshape((2, 1))
    alpha = np.zeros(1)
    C = np.array([[0]])
    atminus1 = np.ones(1)
    Kinv = np.array([[1 / kernel.k(x[0], x[0])]])
    for t in range(1, len(x)):
        kt = compute_k(xdict, x[t], kernel)
        at = Kinv.dot(kt)
        ktt = kernel.k(x[t], x[t])
        eps = ktt - kt.dot(at)
        ktminus1 = compute_k(xdict, x[t - 1], kernel)
        deltak_t = ktminus1 - gamma * kt
        dt = r[t - 1] - alpha.T.dot(deltak_t)
        if eps > nu:
            Kinv = augment_Kinv(eps, at, Kinv)
            # xt represents perfectly itself
            at = np.zeros(Kinv.shape[0])
            at[-1] = 1
            # Augment xdict consequently
            xdict = np.concatenate((xdict, x[t].reshape(2, 1)), axis=1)
            ht = np.concatenate((atminus1, -gamma * np.ones(1)))
            deltak_tt = atminus1.dot(ktminus1 - 2 * gamma * kt) + gamma ** 2 * ktt
            ct = ht - np.concatenate((C.dot(deltak_t), np.zeros(1)))
            st = sigma0 ** 2 + deltak_tt - deltak_t.T.dot(C).dot(deltak_t)
            alpha = np.concatenate((alpha, np.zeros(1)))
            C = augment_C(C)
        else:
            ht = atminus1 - gamma * at
            deltak_tt = ht.T.dot(deltak_t)
            ct = ht - C.dot(deltak_t)
            st = sigma0 ** 2 + deltak_tt - deltak_t.T.dot(C).dot(deltak_t)
        alpha = alpha + (ct / st) * dt
        ctcol = ct.reshape((ct.shape[0], 1))
        C = C + (1 / st) * ctcol.dot(ctcol.T)
        atminus1 = at
        # print(t)
        # print(st)
    return xdict, alpha, C


def compute_state_mean_variance(xdict, xnew, alpha, C, kernel):
    """
    Compute estimate mean and variance at a new point from the learnt vector alpha and the learnt matrix C
    """
    kt = compute_k(xdict, xnew, kernel)
    ktt = kernel.k(xnew, xnew)
    vt = np.dot(kt, alpha)
    pt = ktt - kt.dot(C).dot(kt)
    return vt, pt


def mean_variance_matrices(xdict, coords1, coords2, alpha, C, kernel):
    """
    Compute estimate mean and variance on a 2D grid of new points from the learnt vector alpha and the learnt matrix C
    """
    coords1, coords2 = np.meshgrid(coords1, coords2)
    nx = coords1.shape[0]
    ny = coords2.shape[0]
    M = np.zeros((nx, ny))
    S = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            xnew = np.array([coords1[i, j], coords2[i, j]])
            v, p = compute_state_mean_variance(xdict, xnew, alpha, C, kernel)
            M[i, j] = v
            S[i, j] = p
    return M, S





