import numpy as np
import gptd


def form_H(t, gamma):
    H = np.zeros((t-1, t))
    np.fill_diagonal(H, 1)
    np.fill_diagonal(H[:, 1:], - gamma)
    return H


def form_Q(xdict, H, sigma0, kernel):
    K = gptd.compute_K(xdict, kernel)
    t = K.shape[0]
    Qinv = H.dot(K).dot(H.T) + sigma0 ** 2 * np.eye(t - 1)
    return np.linalg.inv(Qinv)


def compute_alpha(H, Q, r):
    return H.T.dot(Q).dot(r)


def compute_C(H, Q):
    return H.T.dot(Q).dot(H)


def get_mean_variance(xdict, xnew, alpha, C, kernel):
    kx = gptd.compute_k(xdict, xnew, kernel)
    kxx = kernel.k(xnew, xnew)
    v = kx.dot(alpha)
    p = kxx - kx.T.dot(C).dot(kx)
    return v, p





















































