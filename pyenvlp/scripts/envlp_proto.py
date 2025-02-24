#!/usr/bin/env python3

import os, sys
import time
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize

np.set_printoptions(linewidth=100)

"""
Resources
---------

* Renvlp: https://cran.r-project.org/web/packages/Renvlp/index.html
* scipy.optimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
* libLBFGS: https://www.chokkan.org/software/liblbfgs/


R to Python
-----------

* eigen        -> np.linalg.eig
* sweep, apply -> np.apply_along_axis
* GE() GE.R    -> gaussian elimination, scipy.linalg.lu(a, permute_l=True)
* qr, qr.Q     -> np.linalg.qr


envMU Optimization Notes
------------------------

Inputs

This is for the case where multi-Y and single-X:

* u  : dimension of the envelope
* U  : betaOLS * (x'y)', where dims (t x k) * (k x t) => (t x t)
       this is expansion of beta by each cov of x'y, looking at
       permutation of each beta coef by each target dimension. Expresses
       all possible covariance of the betas in the target space.
* M  : cov(Y) - U, where dims are also t x t, this is covariance mat
       of the resids from OLS.
* MU : M + U, marginal sample cov of Y. MU = cov(Y), dim (t x t)

For case where single-Y and multi-X (our usecase):

* u  : dimension of the envelope
* U  : (x'y) inv(y'y) * (x'y)', dim (k x k), span of cov(x,y)
* M  : x'x - U, dim (k x k)
* MU : x'x, dim (k x k), effectively cov of X. MU = cov(X)

Initial value calculation

* initial value is a vector when there's only Y is single target
* eigen is used to approximate inverse?
* 4 distinct values are tried, where each eigen vectors of M, MU
  are combined with "midmatrix" to figure out which eigen vector
  has the largest value. This largest eigen vector is then used as
  initial value, which has the same dimension as the number of targets

Considerations

* there's common pattern in the linear alg operations, duplicated code
  can be reduced for more efficiency and clarity.
* the u=1 can be generalized with implementation using u > 2
* BFGS algorithm has faster C impl that might be worth exploiting
* the 4 initial value calculation can be parallelized
* use of gaussian elimination, use of QR decomp

Questions

* approximation of inverse for MU seems to use eigenvectors, eigenvalues
* use of gaussian elimination for computing initial value before the loop
* use of QR decomposition against the solution of the BFGS routine
* objective values are computed using log sum of eigen values
"""


def envMU_u1(M, U, u):
    """
    This routine covers the case when u=1.
    While the function can be better restructured, it largely follows the
    algorithm present in envMU.R, almost line by line. The functions
    that are specific to R has been changed to numpy equivalents.

    This function essentially implements solving for gammahat, gammahat0,
    where the first is projection matrix associated with material information,
    and the second is projection matrix assocaited with immaterial information.

    The function is also broken up into 2 logical parts:

    * compute sensible initial starting value
    * use BFGS quasi-newton algorithm to find "optimal" gammahat, gamma0hat
    """
    maxiter = 100
    ftol = 1e-3

    def _startv(a, sigma=None):
        return a.T @ sigma @ a

    r = M.shape[0]
    MU = M + U
    ev, E = np.linalg.eig(MU)  # eigen_values and eigen_vectors

    invMU = (E * (1 / ev)) @ E.T
    midmatrix = U
    tmp2_MU = np.apply_along_axis(_startv, 0, E, sigma=midmatrix)  # for each col
    largest_idx = np.argsort(tmp2_MU)[-1]
    init = E[:, [largest_idx]]
    IMI = init.T @ M @ init
    IMinvI = init.T @ invMU @ init
    ev1, E1 = np.linalg.eig(IMI) if IMI.ndim > 0 else (IMI, None)
    ev2, E2 = np.linalg.eig(IMinvI) if IMinvI.ndim > 0 else (IMinvI, None)
    obj1 = np.log(ev1).sum() + np.log(ev2).sum()

    invMU2 = (E * (1 / np.sqrt(ev))) @ E.T
    midmatrix = invMU2 @ U @ invMU2.T
    tmp2_MU = np.apply_along_axis(_startv, 0, E, sigma=midmatrix)
    largest_idx = np.argsort(tmp2_MU)[-1]
    init_MU = E[:, [largest_idx]]
    IMI = init_MU.T @ M @ init_MU
    IMinvI = init_MU.T @ invMU @ init_MU
    ev1, E1 = np.linalg.eig(IMI) if IMI.ndim > 0 else (IMI, None)
    ev2, E2 = np.linalg.eig(IMinvI) if IMinvI.ndim > 0 else (IMinvI, None)
    obj2 = np.log(ev1).sum() + np.log(ev2).sum()
    if obj2 < obj1:
        init = init_MU
        obj1 = obj2

    tmpev, tmpE = np.linalg.eig(M)
    midmatrix = U
    tmp2_M = np.apply_along_axis(_startv, 0, tmpE, sigma=midmatrix)
    largest_idx = np.argsort(tmp2_M)[-1]
    init_M = tmpE[:, [largest_idx]]
    IMI = init_M.T @ M @ init_M
    IMinvI = init_M.T @ invMU @ init_M
    ev1, E1 = np.linalg.eig(IMI) if IMI.ndim > 0 else (IMI, None)
    ev2, E2 = np.linalg.eig(IMinvI) if IMinvI.ndim > 0 else (IMinvI, None)
    obj3 = np.log(ev1).sum() + np.log(ev2).sum()
    if obj3 < obj1:
        init = init_M
        obj1 = obj3

    invM2 = (tmpE * np.sqrt(1 / tmpev)) @ tmpE.T
    midmatrix = invM2 @ U @ invM2.T
    tmp2_M = np.apply_along_axis(_startv, 0, tmpE, sigma=midmatrix)
    largest_idx = np.argsort(tmp2_M)[-1]
    init_M = tmpE[:, [largest_idx]]
    IMI = init_M.T @ M @ init_M
    IMinvI = init_M.T @ invMU @ init_M
    ev1, E1 = np.linalg.eig(IMI) if IMI.ndim > 0 else (IMI, None)
    ev2, E2 = np.linalg.eig(IMinvI) if IMinvI.ndim > 0 else (IMinvI, None)
    obj4 = np.log(ev1).sum() + np.log(ev2).sum()
    if obj4 < obj1:
        init = init_M
        obj1 = obj4

    # gaussian elimination
    ge_pl, ge_u = scipy.linalg.lu(init, permute_l=True)
    sol = np.linalg.solve(ge_u, np.ones_like(ge_u))
    Ginit = init @ sol

    # now iterate and solve
    i = 0

    def objfunc(x, M, invMU):
        f1 = x.T @ x
        f2 = x.T @ M @ x
        f3 = x.T @ invMU @ x
        d1 = x / f1
        d2 = (M @ x) / f2
        d3 = (invMU @ x) / f3
        # we compute both function val and gradient val
        fval = -2 * np.log(f1) + np.log(f2) + np.log(f3)
        gval = -2 * d1 + d2 + d3
        return fval

    xx = None
    objval = obj1
    while i < maxiter:
        # scipy.optimize doesn't need separate grad func
        res = scipy.optimize.minimize(objfunc, Ginit, args=(M, invMU), method="BFGS")
        xx = res.x if res.x.ndim > 1 else res.x[:, np.newaxis]
        gammahat, qrr = np.linalg.qr(xx)
        IMI = gammahat.T @ M @ gammahat
        IMinvI = gammahat.T @ invMU @ gammahat
        ev1, E1 = np.linalg.eig(IMI) if IMI.ndim > 0 else (IMI, None)
        ev2, E2 = np.linalg.eig(IMinvI) if IMinvI.ndim > 0 else (IMinvI, None)
        objval = np.log(ev1).sum() + np.log(ev2).sum()
        if abs(obj1 - objval) < ftol * abs(obj1):
            break
        else:
            obj1 = objval
            i += 1

    gamma0hat = np.linalg.qr(xx, mode="complete")[0][:, u:r]
    objval = objval + np.log(ev).sum()

    return {"gammahat": gammahat, "gamma0hat": gamma0hat, "objfun": objval}


def calc_objective(gx, sigma1, sigma2):
    f1 = gx.T @ sigma1 @ gx  # e.g: x' M x
    f2 = gx.T @ sigma2 @ gx  # e.g.: x' MUinv x
    ev1, E1 = np.linalg.eig(f1) if f1.ndim > 0 else (f1, None)
    ev2, E2 = np.linalg.eig(f2) if f2.ndim > 0 else (f2, None)
    objval = np.log(ev1).sum() + np.log(ev2).sum()
    return objval


def get_initial(E, sigma, nidx=1):
    m, n = E.shape
    # compute "variance" value per eigenvector
    esvals = np.repeat(0, n)
    for j in range(n):
        Evec = E[:, j]
        esvals[j] = Evec.T @ sigma @ Evec.T
    # descending sort index
    sortidx = np.argsort(esvals)[::-1]
    # columns with the top-n largest esvals
    init = E[:, sortidx[:nidx]]
    return init


def envMU_u1_simple(M, U, u):
    """
    Simpler restructured version of the function envMU_u1.
    Produces the same results.
    """
    maxiter = 100
    ftol = 1e-3
    r = M.shape[0]
    # approximated MU inverse
    MU = M + U
    ev, E = np.linalg.eig(MU)
    mev, mE = np.linalg.eig(M)
    invMU = (E * (1 / ev)) @ E.T
    invMU2 = (E * (1 / np.sqrt(ev))) @ E.T
    invM2 = (mE * np.sqrt(1 / mev)) @ mE.T
    # loop through these set to find the minimum value
    init, obj0 = None, np.inf
    for exp, mid, sigma1, sigma2 in [
        (E, U, M, invMU),
        (E, invMU2 @ U @ invMU2.T, M, invMU),
        (mE, U, M, invMU),
        (mE, invM2 @ U @ invM2.T, M, invMU),
    ]:
        init_tmp = get_initial(exp, mid, nidx=u)
        obj_tmp = calc_objective(init_tmp, sigma1, sigma2)
        if obj_tmp < obj0:
            init = init_tmp
            obj0 = obj_tmp

    # gaussian elimination
    ge_pl, ge_u = scipy.linalg.lu(init, permute_l=True)
    sol = np.linalg.solve(ge_u, np.ones_like(ge_u))
    Ginit = init @ sol

    # now iterate and solve
    i = 0

    def objfunc(x, M, invMU):
        f1 = x.T @ x
        f2 = x.T @ M @ x
        f3 = x.T @ invMU @ x
        d1 = x / f1
        d2 = (M @ x) / f2
        d3 = (invMU @ x) / f3
        # we compute both function val and gradient val
        fval = -2 * np.log(f1) + np.log(f2) + np.log(f3)
        gval = -2 * d1 + d2 + d3
        return fval

    xx = None
    objval = obj0
    while i < maxiter:
        # scipy.optimize doesn't need separate grad func
        res = scipy.optimize.minimize(objfunc, Ginit, args=(M, invMU), method="BFGS")
        xx = res.x if res.x.ndim > 1 else res.x[:, np.newaxis]
        # qr decomp
        gammahat, qrr = np.linalg.qr(xx)
        objval = calc_objective(gammahat, M, invMU)
        if abs(obj0 - objval) < ftol * abs(obj0):
            break
        else:
            obj0 = objval
            i += 1

    gamma0hat = np.linalg.qr(xx, mode="complete")[0][:, u:r]
    objval = objval + np.log(ev).sum()

    return {"gammahat": gammahat, "gamma0hat": gamma0hat, "objfun": objval}


def GE(Aog):
    """
    Gaussian elimination, returns the index of linear equations that
    needs to be solved in order. Requires the modification of A matrix,
    so  makes an internal copy.
    """
    A = Aog.copy()
    n, p = A.shape
    idx = np.repeat(0, p)
    res_idx = np.arange(n)
    i = 0
    while i < p:
        tmp = np.max(np.abs(A[res_idx, i]))
        stmp = np.setdiff1d(np.where(np.abs(A[:, i]) == tmp), idx)
        idx[i] = stmp[0]
        res_idx = np.setdiff1d(res_idx, idx[i])
        for j in range(n - 2):
            A[res_idx[j], :] = (
                A[res_idx[j], :] - A[res_idx[j], i] / A[idx[i], i] * A[idx[i], :]
            )
        i += 1
    return np.concatenate([idx, res_idx])


def envMU_un(M, U, u):
    """
    Covers more general case when u > 2. Uses coordinate descent combined with
    BFGS optimizer to solve for gammahat, gammahat0.
    """
    maxiter = 100
    ftol = 1e-3
    r = M.shape[0]

    # approximated MU inverse
    MU = M + U
    ev, E = np.linalg.eig(MU)
    mev, mE = np.linalg.eig(M)
    invMU = (E * (1 / ev)) @ E.T
    invMU2 = (E * (1 / np.sqrt(ev))) @ E.T
    invM2 = (mE * np.sqrt(1 / mev)) @ mE.T

    # loop through these permutations to determine
    # the minimum value
    init, obj0 = None, np.inf
    for exp, mid, sigma1, sigma2 in [
        (E, U, M, invMU),
        (E, invMU2 @ U @ invMU2.T, M, invMU),
        (mE, U, M, invMU),
        (mE, invM2 @ U @ invM2.T, M, invMU),
    ]:
        init_tmp = get_initial(exp, mid, nidx=u)
        obj_tmp = calc_objective(init_tmp, sigma1, sigma2)
        if obj_tmp < obj0:
            init = init_tmp
            obj0 = obj_tmp

    # gaussian elimination
    geidx = GE(init)
    initu = init[geidx[:u], :]
    sol = np.linalg.solve(initu, np.eye(initu.shape[0]))
    Ginit = init @ sol

    GUG = Ginit.T @ (M @ Ginit)
    GVG = Ginit.T @ (invMU @ Ginit)
    # now iterate and solve
    i = 0

    initv = Ginit[geidx[u:r], :]
    t4 = (initv.T @ initv) + np.eye(u)  # dim: u x u

    def objfunc(x, t2, t3, invt4, invC1, invC2, Mj, invMUj):
        tmp2 = x + t2
        tmp3 = x + t3
        f1 = invt4 @ x
        f2 = invC1 @ tmp2
        f3 = invC2 @ tmp3
        # if x @ f1 < -1:
        #     print(f"x @ f1 = {x @ f1} is less than 1!")
        fval = (
            -2 * np.log(1 + (x @ f1))
            + np.log(1 + Mj * (tmp2.T @ f2))
            + np.log(1 + invMUj * (tmp3.T @ f3))
        )
        grad = (
            -4 * f1 / (1 + (x @ f1))
            + 2 * f2 / (1 / Mj + (tmp2.T @ f2))
            + 2 * f3 / (1 / invMUj + (tmp3.T @ f3))
        )
        return fval

    def _cholinv_sci(A):
        L = scipy.linalg.cholesky(A, lower=True)
        Ainv = scipy.linalg.cho_solve((L, True), np.eye(A.shape[0]))
        return Ainv

    def _cholinv(A):
        L = np.linalg.cholesky(A)
        Linv = np.linalg.solve(L, np.eye(L.shape[0]))
        Ainv = Linv.T @ np.eye(Linv.shape[0]) @ Linv
        return Ainv

    xx = None
    objval = obj0
    while i < maxiter:
        # coordinate descent across rows
        for j in geidx[u:r]:
            g = Ginit[[j], :]

            # matrices with j-th row removed
            Ginit_rest = np.delete(Ginit, [j], axis=0)
            M_rest = np.delete(M, [j], axis=0)[:, j]
            invMU_rest = np.delete(invMU, [j], axis=0)[:, j]

            t2 = (Ginit_rest.T @ M_rest) / M[j, j]
            t3 = (Ginit_rest.T @ invMU_rest) / invMU[j, j]

            # GUGt2.T @ GUGt2 is expansion
            GUGt2 = g + t2
            GUG = GUG - (GUGt2.T @ GUGt2) * M[j, j]

            GVGt2 = g + t3
            GVG = GVG - (GVGt2.T @ GVGt2) * invMU[j, j]

            t4 = t4 - (g.T @ g)  # expansion, note the minus here

            invC1 = _cholinv(GUG)  # better than pinv
            invC2 = _cholinv(GVG)
            invt4 = _cholinv(t4)

            # scipy.optimize doesn't need separate grad func
            res = scipy.optimize.minimize(
                objfunc,
                Ginit[j, :],
                args=(t2, t3, invt4, invC1, invC2, M[j, j], invMU[j, j]),
                method="BFGS",
            )
            xx = res.x
            Ginit[j, :] = xx

            g = Ginit[[j], :]

            GUGt2 = g + t2
            GUG = GUG + (GUGt2.T @ GUGt2) * M[j, j]

            GVGt2 = g + t3
            GVG = GVG + (GVGt2.T @ GVGt2) * invMU[j, j]

            t4 = t4 + g.T @ g  # NOTE: plus here

        # qr decomp
        gammahat, qrr = np.linalg.qr(Ginit)
        objval = calc_objective(gammahat, M, invMU)
        if abs(obj0 - objval) < ftol * abs(obj0):
            break
        else:
            obj0 = objval
            i += 1

    gamma0hat = np.linalg.qr(Ginit, mode="complete")[0][:, u:r]
    objval = objval + np.log(ev).sum()

    return {"gammahat": gammahat, "gamma0hat": gamma0hat, "objfun": objval}


def main():
    dataroot = "/dat/pm_nverses/dev/research/experimental/envelope"
    testdata = f"{dataroot}/tmp/wheat.pq"
    df = pd.read_parquet(testdata)
    X = df.X.values
    Y = df.drop("X", axis=1).values
    n = Y.shape[0]
    r = Y.shape[1]
    nadj = (n - 1) / n
    sigY = np.cov(Y.T) * nadj
    sigYX = np.cov(Y.T, X)[-1][:-1] * nadj
    sigX = np.cov(X) * nadj

    betaOLS = sigYX / sigX if sigX.ndim == 0 else np.linalg.solve(sigX, sigYX)
    print(betaOLS)
    # now ready for checking against envMU
    # U = betaOLS @ sigYX

    # note this expansion
    # U = beta * x'y = (x'x)^-1 x'y * x'y
    U = betaOLS[:, np.newaxis] @ sigYX[:, np.newaxis].T
    M = sigY - U
    u = 1
    print(U)
    print(M)
    res = envMU_u1_simple(M, U, u)
    for k, v in res.items():
        print(f"\n{k}\n{v}")

    t0 = time.time()
    res2 = envMU_un(M, U, 2)
    for k, v in res2.items():
        print(f"\n{k}\n{v}")
    print(f"envMU_un took: {time.time()-t0:.4f} secs")


if __name__ == "__main__":
    main()
