import numpy as np
import scipy.stats as stats


class Gaussian_Kernel:

    def __init__(self, sigma):
        self.sigma = sigma

    def k(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.sigma ** 2))
        # return stats.norm(0, self.sigma).pdf(np.linalg.norm(x1 - x2) ** 2)


def compute_k(xdict, xnew, kernel):
    n = xdict.shape[1]
    k = np.array([kernel.k(xdict[:, i], xnew) for i in range(n)])
    return k


def compute_K(xdict, kernel):
    n = xdict.shape[1]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = kernel.k(xdict[:, i], xdict[:, j])
            K[j, i] = K[i, j]
    return K


def initialize_H(gamma):
    return np.array([[1, -gamma]])


def intialize_Q(K, H, sigma0):
    HTKH = H.dot(K).dot(H.T) + sigma0 * np.eye(1)
    return np.linalg.inv(HTKH)


def initialize_alpha(H, Q, r):
    return H.T.dot(Q)[:, 0]


def initialize_C(H, Q):
    return H.T.dot(Q).dot(H)


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
    Kinv_augmented[-1, 0:nminus1] = - a.T
    Kinv_augmented[0:nminus1, -1] = - a
    return (1 / epsilon) * Kinv_augmented


def augment_A_case1(A, a):
    nminus1 = A.shape[0]
    acol = a.reshape((nminus1, 1))
    return np.concatenate((A, acol.T), axis=0)


def compute_h(atminus1, at, gamma):
    return atminus1 - gamma * at


def augment_H_case1(H, atminus1, at, gamma):
    h = compute_h(atminus1, at, gamma)
    h = h.reshape((1, h.shape[0]))
    H_augmented = np.concatenate((H, h), axis=0)
    return H_augmented


def compute_g(Qtminus1, Htminus1, deltak_tminus1):
    return Qtminus1.dot(Htminus1.dot(deltak_tminus1))


def compute_c(Htminus1, gt, ht):
    # TODO : Is there a forgiven transpose sign for H in the definition of c in the paper ?
    # return np.dot(Htminus1, gt) - ht
    return np.dot(Htminus1.T, gt) - ht


def compute_s(sigma, ct, deltak_tminus1):
    return sigma ** 2 - np.dot(ct.T, deltak_tminus1)


def augment_Q(Q, s, g):
    nminus1 = Q.shape[0]
    gcol = g.reshape((g.shape[0], 1))
    first_block = s * Q + np.dot(gcol, gcol.T)
    Q_augmented = np.zeros((nminus1 + 1, nminus1 + 1))
    Q_augmented[: nminus1, : nminus1] = first_block
    Q_augmented[-1, 0:nminus1] = g.T
    Q_augmented[0: nminus1, -1] = g
    Q_augmented[-1, -1] = 1
    return Q_augmented


def update_alpha_case1(alpha_tminus1, deltak_tminus1, ct, st, rtminus1):
    return alpha_tminus1 + (ct / st) * \
        (np.dot(deltak_tminus1.T, alpha_tminus1) - rtminus1)


def update_C_case1(Ctminus1, ct, st):
    ccol = ct.reshape((ct.shape[0], 1))
    return Ctminus1 + (1 / st) * np.dot(ccol, ccol.T)


def augment_H_case2(H, atminus1, gamma):
    rh, ch = H.shape[0], H.shape[1]
    H_augmented = np.zeros((rh + 1, ch + 1))
    H_augmented[0: rh, 0: ch] = H
    H_augmented[-1, :ch] = atminus1
    H_augmented[:rh, -1] = np.zeros(rh)
    H_augmented[-1, -1] = - gamma
    return H_augmented


def compute_deltak_tt(atminus1, deltak_tminus1, kt, ktt, gamma):
    return np.dot(atminus1.T, deltak_tminus1 - gamma * kt) + (gamma ** 2) * ktt


def compute_hat_s(atminus1, kt, ktt, sigma0, deltak_tminus1, Ctminus1, gamma):
    deltak_tt = compute_deltak_tt(atminus1, deltak_tminus1, kt, ktt, gamma)
    return sigma0 ** 2 + deltak_tt - \
        deltak_tminus1.dot(Ctminus1.dot(deltak_tminus1))


def compute_hat_c(Htminus1, gt, atminus1):
    # TODO : Is there a forgiven transpose sign in the definition of hat_c in the paper ?
    # return np.dot(Htminus1, gt) - atminus1
    return np.dot(Htminus1.T, gt) - atminus1


def update_alpha_case2(alpha_tminus1,
                       deltak_tminus1,
                       hat_ct,
                       hat_st,
                       rtminus1,
                       gamma):
    nminus1 = alpha_tminus1.shape[0]
    alpha_t = np.zeros(nminus1 + 1)
    alpha_t[: nminus1] = update_alpha_case1(
        alpha_tminus1, deltak_tminus1, hat_ct, hat_st, rtminus1)
    alpha_t[-1] = (gamma / hat_st) * \
        (np.dot(deltak_tminus1, alpha_tminus1) - rtminus1)
    return alpha_t


def update_C_case2(Ctminus1, hat_ct, hat_st, gamma):
    nminus1 = Ctminus1.shape[0]
    hat_ct_col = hat_ct.reshape((nminus1, 1))
    Ct = np.zeros((nminus1 + 1, nminus1 + 1))
    Ct[: nminus1, : nminus1] = update_C_case1(Ctminus1, hat_ct, hat_st)
    Ct[-1, 0:nminus1] = (gamma / hat_st) * hat_ct.T
    Ct[0: nminus1, -1] = (gamma / hat_st) * hat_ct
    Ct[-1, -1] = gamma ** 2 / hat_st
    return Ct


def first_step(x0, x1, r0, gamma, sigma0, kernel):
    # Dict initialization
    xdict = np.array(x0).reshape((2, 1))
    Kinv = np.array([[1 / kernel.k(x0, x0)]])
    ktt = kernel.k(x1, x1)
    kt = compute_k(xdict, x1, kernel)
    # Compute epsilon
    ahat_t = compute_a(kt, Kinv)
    eps = compute_epsilon(ktt, kt, ahat_t)
    # For simplicity's sake, no test for the first step, x1 is automatically
    # added to the dictionnary
    xdict = np.concatenate((xdict, x1.reshape((2, 1))), axis=1)
    Kinv = augment_Kinv(eps, ahat_t, Kinv)
    # We initialize H instead of updating it for the first augmentation step
    H = initialize_H(gamma)
    # We intialize Q instead of updating it for the first augmentation step
    # For that we need to create the kernel matrix
    K = compute_K(xdict, kernel)
    Q = intialize_Q(K, H, sigma0)
    # At first step we compute alpha directly since we cannot define the
    # recursions
    alpha = initialize_alpha(H, Q, np.array([r0]))
    # At first step we compute C directly since we cannot define the recursions
    C = initialize_C(H, Q)
    at = np.array([0, 1])
    return xdict, Kinv, H, Q, ahat_t, at, alpha, C


def test_phase(xdict, xnew, Kinv, nu, kernel):
    ktt = kernel.k(xnew, xnew)
    kt = compute_k(xdict, xnew, kernel)
    # Compute epsilon for test
    at = compute_a(kt, Kinv)
    eps = compute_epsilon(ktt, kt, at)
    return (eps > nu), at, kt, eps


def iterate_case1(xdict,
                  rtminus1,
                  atminus1,
                  at,
                  Htminus1,
                  Qtminus1,
                  kt,
                  alpha_tminus1,
                  Ctminus1,
                  gamma,
                  sigma0,
                  kernel):
    Ht = augment_H_case1(Htminus1, atminus1, at, gamma)
    ht = compute_h(atminus1, at, gamma)
    deltak_tminus1 = compute_k(xdict, xdict[:, -1], kernel) - gamma * kt
    # gt = Qtminus1.dot(Htminus1.dot(deltak_tminus1))
    gt = compute_g(Qtminus1, Htminus1, deltak_tminus1)
    ct = compute_c(Htminus1, gt, ht)
    st = compute_s(sigma0, ct, deltak_tminus1)
    Qt = augment_Q(Qtminus1, st, gt)
    alpha_t = update_alpha_case1(alpha_tminus1,
                                 deltak_tminus1,
                                 ct,
                                 st,
                                 rtminus1)
    Ct = update_C_case1(Ctminus1, ct, st)
    return Ht, Qt, alpha_t, Ct


def iterate_case2(xdict,
                  xnew,
                  rtminus1,
                  atminus1,
                  ahat_t,
                  epst,
                  Kinv,
                  Htminus1,
                  Qtminus1,
                  kt,
                  alpha_tminus1,
                  Ctminus1,
                  gamma,
                  sigma0,
                  kernel):
    Kinv = augment_Kinv(epst, ahat_t, Kinv)
    Ht = augment_H_case2(Htminus1, atminus1, gamma)
    deltak_tminus1 = compute_k(xdict, xdict[:, -1], kernel) - gamma * kt
    gt = compute_g(Qtminus1, Htminus1, deltak_tminus1)
    st = compute_hat_s(atminus1,
                       kt,
                       kernel.k(xnew, xnew),
                       sigma0,
                       deltak_tminus1,
                       Ctminus1,
                       gamma)
    ct = compute_hat_c(Htminus1, gt, atminus1)
    alpha_t = update_alpha_case2(alpha_tminus1,
                                 deltak_tminus1,
                                 ct,
                                 st,
                                 rtminus1,
                                 gamma)
    Ct = update_C_case2(Ctminus1, ct, st, gamma)
    Qt = augment_Q(Qtminus1, st, gt)
    xdict = np.concatenate((xdict, xnew.reshape((2, 1))), axis=1)
    return xdict, Kinv, Ht, Qt, alpha_t, Ct
