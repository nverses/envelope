import time
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
from .lau import geindex, cholinv

USE_CPP_ENVOPT = False
try:
    from ._pyenvlp import EnvOpt

    USE_CPP_ENVOPT = True
except Exception:
    pass


def envMU(M, U, u, initial=None):
    """
    Tries faster cpp version if available, else uses python version.
    Python ver is retained for reference and for educational purposes.
    """
    if USE_CPP_ENVOPT:
        return EnvOpt.solve_envelope(M, U, u, initial)
    else:
        return envMUpy(M, U, u, initial)


def _calc_objective(gx, sigma1, sigma2):
    f1 = gx.T @ sigma1 @ gx  # e.g: x' M x
    f2 = gx.T @ sigma2 @ gx  # e.g.: x' MUinv x
    ev1, E1 = np.linalg.eig(f1) if f1.ndim > 0 else (f1, None)
    ev2, E2 = np.linalg.eig(f2) if f2.ndim > 0 else (f2, None)
    objval = np.log(ev1).sum() + np.log(ev2).sum()
    return objval


def _pick_initial(E, sigma, nidx=1):
    """
    Pick subset of eigenvectors that best minimizes:
        eval_j = E_j @ sigma @ E_j.T
    """
    m, n = E.shape
    # compute "variance" value per eigenvector
    esvals = np.repeat(0.0, n).astype(np.float64)
    # also could do: esvals = np.apply_along_axis(startv, 1, E)
    for j in range(n):
        Evec = E[:, j]
        esvals[j] = Evec.T @ sigma @ Evec
    # descending sort index
    sortidx = np.argsort(esvals)[::-1]
    # columns with the top-n largest quadratic value
    init = E[:, sortidx[:nidx]]
    return init


def _compute_initial(M, U, u=1, init=None):
    """
    The initial vector (starting point) is essentially subset 'u'
    eigenvectors that best minimizes the Sigma (M, invMU, etc..).
    As such, the eigenvector signs can change from 1 vector to
    the next. Later in the code, these eigenvector subset matrix
    is put through gaussian elimination and then "normalized" back
    out into Ginit matrix.
    """
    MU = M + U
    ev, E = np.linalg.eig(MU)
    mev, mE = np.linalg.eig(M)
    # approximated MU inverse
    invMU = (E * (1 / ev)) @ E.T
    invMU2 = (E * (1 / np.sqrt(ev))) @ E.T
    invM2 = (mE * np.sqrt(1 / mev)) @ mE.T
    obj0 = np.inf
    # determine the initial value
    if init is not None:
        obj0 = _calc_objective(init, M, invMU)
    else:
        # loop through these set to find the minimum value
        for exp, mid, sigma1, sigma2 in [
            (E, U, M, invMU),
            (E, invMU2 @ U @ invMU2.T, M, invMU),
            (mE, U, M, invMU),
            (mE, invM2 @ U @ invM2.T, M, invMU),
        ]:
            init_tmp = _pick_initial(exp, mid, nidx=u)
            obj_tmp = _calc_objective(init_tmp, sigma1, sigma2)
            if obj_tmp < obj0:
                init = init_tmp
                obj0 = obj_tmp
    # return the initial value along with other values we need
    return init, obj0, ev, invMU


def _envMU_u1(M, U, init=None):
    """
    Covers the case when u=1
    """
    maxiter = 100
    ftol = 1e-3
    r = M.shape[0]

    # get initial value
    init, obj0, ev, invMU = _compute_initial(M, U, 1, init)

    # gaussian elimination
    ge_pl, ge_u = scipy.linalg.lu(init, permute_l=True)
    sol = np.linalg.solve(ge_u, np.ones_like(ge_u))
    Ginit = init @ sol

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
    # now iterate and solve
    i = 0
    while i < maxiter:
        # scipy.optimize doesn't need separate grad func
        res = scipy.optimize.minimize(
            objfunc,
            Ginit,
            args=(M, invMU),
            method="BFGS",
        )
        xx = res.x if res.x.ndim > 1 else res.x[:, np.newaxis]
        # qr decomp
        gammahat, qrr = np.linalg.qr(xx)
        objval = _calc_objective(gammahat, M, invMU)
        if abs(obj0 - objval) < ftol * abs(obj0):
            break
        else:
            obj0 = objval
            i += 1

    gamma0hat = np.linalg.qr(xx, mode="complete")[0][:, 1:r]
    objval = objval + np.log(ev).sum()

    return {
        "gammahat": gammahat,
        "gamma0hat": gamma0hat,
        "objfun": objval,
        "niter": i,
    }


def _envMU_un(M, U, u, init=None, urange=None):
    """
    Covers more general case when u > 2. Uses coordinate descent
    combined with BFGS optimizer to solve for gammahat, gammahat0.
    """
    maxiter = 100
    ftol = 1e-3
    r = M.shape[0]

    # get initial value
    init, obj0, ev, invMU = _compute_initial(M, U, u, init)

    # gaussian elimination
    geidx = geindex(init)
    initu = init[geidx[:u], :]
    sol = np.linalg.solve(initu, np.eye(initu.shape[0]))
    Ginit = init @ sol

    GUG = Ginit.T @ (M @ Ginit)
    GVG = Ginit.T @ (invMU @ Ginit)

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

    xx = None
    objval = obj0
    # define coordinate descent range [u,r], unless we have specific range
    jvec = geidx[u:r] if urange is None else geidx[urange]
    # now iterate and solve
    toti = i = 0
    while i < maxiter:
        # coordinate descent across rows
        for j in jvec:
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

            invC1 = cholinv(GUG)  # better than pinv
            invC2 = cholinv(GVG)
            invt4 = cholinv(t4)

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
            toti += 1

        # qr decomp
        gammahat, qrr = np.linalg.qr(Ginit)
        objval = _calc_objective(gammahat, M, invMU)
        if abs(obj0 - objval) < ftol * abs(obj0):
            break
        else:
            obj0 = objval
            i += 1

    gamma0hat = np.linalg.qr(Ginit, mode="complete")[0][:, u:r]
    objval = objval + np.log(ev).sum()

    return {
        "gammahat": gammahat,
        "gamma0hat": gamma0hat,
        "objfun": objval,
        "niter": toti,  # outer iter x coord descent iter
    }


def _envMU_ur(M, U, u, init=None):
    """
    Covers case u = r - 1, which is same as u = n where coordinate
    descent happens only for 1 iteration, so we call the function
    that generalizes this.
    """
    # causes coordinate once for equation at r-th row (last index)
    urange = np.array([M.shape[0] - 1])
    return _envMU_un(M, U, u, init=init, urange=urange)


def envMUpy(M, U, u, initial=None):
    """
    General entry point for M, U optimization.
    """
    # input validation
    if M.ndim < 2 or U.ndim < 2:
        raise ValueError("matrices M, U contains less than 2 dimensions.")
    dimM = M.shape
    dimU = U.shape
    r = dimM[0]
    if dimM[0] != dimM[1] or dimU[0] != dimU[1]:
        raise ValueError("M and U should be square matrices.")
    if dimM[0] != dimU[0]:
        raise ValueError("M and U should have the same dimension.")
    if np.linalg.matrix_rank(M) < r:
        raise ValueError("M should be positive definite.")
    if u > r or u < 0:
        raise ValueError("u should be between 0 and r.")

    # specialization based on value of u
    out = {"gammahat": None, "gamma0hat": np.eye(r), "objfun": np.nan}

    t0 = time.time()
    # u = 0     no material info
    # u = r     all dimensions material
    # u = 1     single dim matters
    # u = r-1   all dim matters except one
    # u = n     material dim is subset
    if u == 0:
        out["gammahat"] = None
        out["gamma0hat"] = np.eye(r)
        ev, E = np.linalg.eig(M + U)
        out["objfun"] = np.log(ev).sum()
    elif u == r:
        out["gammahat"] = np.eye(r)
        out["gamma0hat"] = None
        ev, E = np.linalg.eig(M)
        out["objfun"] = np.log(ev).sum()
    elif u == 1:
        out = _envMU_u1(M, U, init=initial)
    elif u == r - 1 and u != 1:
        out = _envMU_ur(M, U, u, init=initial)
    else:
        out = _envMU_un(M, U, u, init=initial)

    out["opt_elapsed"] = time.time() - t0
    return out
