import numpy as np
import scipy.linalg


def cov(a, b=None):
    n = a.shape[0]
    amu = a - a.mean(axis=0)
    if b is None:
        return amu.T @ amu / (n - 1)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"cov: dimension for a and b does not match")
    bmu = b - b.mean(axis=0)
    return amu.T @ bmu / (n - 1)


def cholinv_sci(A):
    L = scipy.linalg.cholesky(A, lower=True)
    Ainv = scipy.linalg.cho_solve((L, True), np.eye(A.shape[0]))
    return Ainv


def cholinv(A):
    L = np.linalg.cholesky(A)
    Linv = np.linalg.solve(L, np.eye(L.shape[0]))
    Ainv = Linv.T @ np.eye(Linv.shape[0]) @ Linv
    return Ainv


def geindex(Aog):
    """
    Gaussian elimination, returns the index of linear equations that
    needs to be solved in order. Requires the modification of A matrix,
    so makes an internal copy.
    """
    A = Aog.copy()
    n, p = A.shape
    idx = np.repeat(0, p)
    res_idx = np.arange(n)
    i = 0
    while i < p:
        tmp = np.max(np.abs(A[res_idx, i]))
        stmp = np.setdiff1d(np.where(np.abs(A[:, i]) == tmp), idx)
        if len(stmp) > 0:
            idx[i] = stmp[0]
        res_idx = np.setdiff1d(res_idx, idx[i])
        for j in range(n - i - 1):
            A[res_idx[j], :] = (
                A[res_idx[j], :] - A[res_idx[j], i] / A[idx[i], i] * A[idx[i], :]
            )
        i += 1
    return np.concatenate([idx, res_idx])
