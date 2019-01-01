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


def compute_delta_k(ktminus1, kt, gamma):
    return ktminus1 - gamma * kt


def augment_Kinv(epsilon, a, Kinv):
    nminus1 = Kinv.shape[0]
    acol = a.reshape((a.shape[0], 1))
    aaT = np.dot(acol, acol.T)
    Kinv_augmented = np.zeros((nminus1 + 1, nminus1 + 1))
    firstblock = epsilon * Kinv + aaT
    Kinv_augmented[0:nminus1, 0:nminus1] = firstblock
    Kinv_augmented[-1, -1] = 1
    Kinv_augmented[-1, 0:nminus1] = acol.T
    Kinv_augmented[0:nminus1, -1] = acol
    return (1 / epsilon) * Kinv_augmented


def compute_h(atminus1, at, gamma):
    return atminus1 - gamma * at


def augment_H(H, atminus1, at, gamma):
    h = compute_h(atminus1, at, gamma)
    H_augmented = np.concatenate((H, h.T), axis=0)
    return H_augmented


def compute_g(Qtminus1, Htminus1, deltak_tminus1):
    return Qtminus1.dot(Htminus1.dot(deltak_tminus1))


def compute_c(Htminus1, gt, ht):
    return np.dot(Htminus1, gt) - ht


def compute_s(sigma, ct, deltak_tminus1):
    return sigma ** 2 - np.dot(ct.T, deltak_tminus1)


def augment_Q(Q, s, g):
    nminus1 = Q.shape[0]
    gcol = g.reshape((g.shape[0], 1))
    first_block = s * Q + np.dot(gcol, gcol.T)
    Q_augmented = np.zeros((nminus1 + 1, nminus1 + 1))
    Q_augmented[: nminus1, : nminus1] = first_block
    Q_augmented[-1, 0:nminus1] = gcol.T
    Q_augmented[0: nminus1, -1] = gcol
    Q_augmented[-1, -1] = 1


def update_alpha_case1(alpha_tminus1, deltak_tminus1, ct, st, rtminus1):
    return alpha_tminus1 + (ct / st) * (np.dot(deltak_tminus1.T, alpha_tminus1) - rtminus1)


def update_C_case1(Ctminus1, ct, st):
    ccol = ct.reshape((c.shape[0], 1))
    return Ctminus1 + (1 / st) * np.dot(ccol, ccol.T)


def compute_deltak_tt(atminus1, deltak_tminus1, kt, ktt, gamma):
    return np.dot(atminus1.T, deltak_tminus1 - gamma * kt) + (gamma ** 2) * ktt


def compute_hat_s(atminus1, kt, ktt, sigma, deltak_tminus1, Ctminus1, gamma):
    deltak_tt = compute_deltak_tt(atminus1, deltak_tminus1, kt, ktt, gamma)
    return sigma ** 2 + deltak_tt - deltak_tminus1.dot(Ctminus1.dot(deltak_tminus1))


def compute_hat_c(Htminus1, gt, atminus1):
    return np.dot(Htminus1, gt) - atminus1


def update_alpha_case2(alpha_tminus1, deltak_tminus1, hat_ct, hat_st, rtminus1, gamma):
    nminus1 = alpha_tminus1.shape[0]
    alpha_t = np.zeros(nminus1 + 1)
    alpha_t[: nminus1] = update_alpha_case1(alpha_tminus1, deltak_tminus1, hat_ct, hat_st, rtminus1)
    alpha_t[-1] = (gamma / hat_st) * (np.dot(deltak_tminus1, alpha_tminus1) - rtminus1)
    return alpha_t


def update_C_case2(Ctminus1, hat_ct, hat_st, gamma):
    nminus1 = Ctminus1.shape[0]
    hat_ct_col = hat_ct.reshape((nminus1, 1))
    Ct = np.zeros((nminus1 + 1, nminus1 + 1))
    Ct[: nminus1, : nminus1] = update_C_case1(Ctminus1, hat_ct, hat_st)
    Ct[-1, 0:nminus1] = (gamma / hat_st) * hat_ct_col.T
    Ct[0: nminus1, -1] = (gamma / hat_st) * hat_ct_col
    Ct[-1, -1] = gamma ** 2 / hat_st
    return Ct


kernel = Gaussian_Kernel(1)

xdict = np.array([[2, 3, 4], [0, 1, 2]])
xnew = np.array([1, 1])

test = compute_k(xdict, xnew, kernel)