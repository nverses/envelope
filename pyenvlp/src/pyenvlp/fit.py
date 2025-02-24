import numpy as np
from .lau import cov, cholinv
from .opt import envMU


def fit_yenv_gram(sigY, sigYX, sigX, u, init=None):
    """
    Fit envelope model using grammian matrices cov(y, y), cov(x, y), cov(x, x).
    Called by top-level function fit_yenv that uses full matrices X, Y.
    """
    r = sigY.shape[0]
    p = sigX.shape[0] if sigX.ndim > 0 else 1
    if u > r or u < 0:
        raise ValueError("u must be an integer between 0 and r.")
    # check the dim of initial value
    if init is not None:
        if init.shape[0] != r or init.shape[1] != u:
            raise ValueError(f"wrong dimension for init, requires (r={r} x u={u})")

    # compute the OLS solution first
    # print(f"ndim={sigX.ndim}, shape={sigX.shape}")
    # xn = sigX.itemsize / sigX.dtype.itemsize
    betaOLS = sigYX / sigX if p == 1 else np.linalg.solve(sigX, sigYX.T).T
    U = betaOLS @ sigYX.T  # expansion: t x t
    M = sigY - U  # cov of residuals

    # do the optimization
    res = envMU(M, U, u, initial=init)

    # envelope projecteed beta
    gammahat = res["gammahat"]
    # gamma0hat = res["gamma0hat"]
    betahat = None
    if u == 0:
        betahat = np.zeros((r, p))
    elif u == r:
        betahat = betaOLS
    else:
        etahat = gammahat.T @ betaOLS  # reduction to dimension u
        betahat = gammahat @ etahat  # expansion from dimention u

    # additional data to be returned
    res.update(
        {
            "u": u,
            "r": r,
            "p": p,
            "U": U,
            "M": M,
            "betaols": betaOLS,  # cannonical OLS solution
            "beta": betahat,  # envelope solution
        }
    )
    return res


def fit_yenv(X, Y, u, init=None, asy=True):
    """
    Fits envelope model of Y, given envelope subspace dimension u.
    Returns the envelope modified beta, along with other measures
    involved in solving for projection matrix. Y matrix is usually
    multi-target, and X can be a single vector or multi-dimensional.

    Args:
        X : (n x p)
        Y : (n x r)
        u : scalar integer between [0, r]

        omegahat = gammahat.T @ M @ gammahat
        omega0hat = gamma0hat.T @ sigY @ gamma0hat
        sigma1 = gammahat @ omegahat @ gammahat.T
        sigmahat = sigma1 + (gamma0hat @ omega0hat @ gamma0hat.T)
        loglik = -n * r / 2 * (np.log(2 * np.pi) + 1) - n / 2 * objfun
        etahat = gammahat.T @ betaOLS  # reduction to dimension u

    Returns dict with keys:
        gamma       : projection matrix for material info Y|X
        gamma0      : projection matrix for immaterial info
        mu:         : intercept computed from envelope beta
        beta:       : envelope modified OLS beta
        sigma       : total variance (material + immaterial variance)
        eta         : projected ols beta in reduced u dimension
        omega:      : projected resid cov in reduced u dimension (material)
        omega0:     : projected y'y cov in reduced u dimension (immaterial)
        loglik      : log likelihood value
        n           : number of samples
        covMatrix   : final covariance matrix
        asySE       : asymptotic standard error
        ratio       : asymptotic SE of OLS / asymptotic SE of envelope for each beta
        niter       : number of iterations taken in envMU opt
        opt_elapsed : time taken to solve in envMU opt
    """
    n, r = Y.shape
    p = X.shape[1] if X.ndim > 1 else 1
    if n != X.shape[0]:
        raise ValueError(f"X and Y should have the same number of observations.")
    if X.ndim == 1:
        X = X[:, np.newaxis]
    # NOTE: check for duplicate columns here

    nadj = (n - 1) / n
    sigY = cov(Y) * nadj
    sigYX = cov(Y, X) * nadj
    sigX = cov(X) * nadj

    res = fit_yenv_gram(sigY, sigYX, sigX, u, init=init)

    # optimized values
    U = res["U"]
    M = res["M"]
    betaOLS = res["betaols"]
    gammahat = res["gammahat"]
    gamma0hat = res["gamma0hat"]
    objfun = res["objfun"]
    # to be computed below
    betahat = None
    covMatrix = None
    asySE = None
    ratio = None

    if u == 0:
        etahat = None
        omegahat = None
        omega0hat = sigY
        muhat = Y.mean(axis=0)
        betahat = np.zeros((r, p))
        sigmahat = sigY
        loglik = -n * r / 2 * (np.log(2 * np.pi) + 1) - n / 2 * objfun
        if asy:
            ratio = np.ones((r, p))
    elif u == r:
        invsigX = cholinv(sigX)
        etahat = betaOLS
        omegahat = M
        omega0hat = None
        muhat = Y.mean(axis=0) - betaOLS @ X.mean(axis=0)
        betahat = betaOLS
        sigmahat = M
        loglik = -n * r / 2 * (np.log(2 * np.pi) + 1) - n / 2 * objfun
        if asy:
            covMatrix = np.kron(invsigX, M)
            # matrix(sqrt(diag(covMatrix)), nrow = r)
            asySE = np.sqrt(np.diag(covMatrix))
            ratio = np.ones((r, p))
    else:
        invsigX = cholinv(sigX)
        etahat = gammahat.T @ betaOLS  # reduction to dimension u
        betahat = gammahat @ etahat  # expansion from dimention u
        muhat = Y.mean(axis=0) - betahat @ X.mean(axis=0)
        omegahat = gammahat.T @ M @ gammahat
        omega0hat = gamma0hat.T @ sigY @ gamma0hat
        sigma1 = gammahat @ omegahat @ gammahat.T
        sigmahat = sigma1 + (gamma0hat @ omega0hat @ gamma0hat.T)
        loglik = -n * r / 2 * (np.log(2 * np.pi) + 1) - n / 2 * objfun
        if asy:
            covMatrix = np.kron(invsigX, M)  # appears below?
            asyFm = np.sqrt(np.diag(covMatrix))
            invOmegahat = cholinv(omegahat)
            invOmega0hat = cholinv(omega0hat)
            tmp = (
                np.kron(etahat @ sigX @ etahat.T, invOmega0hat)
                + np.kron(invOmegahat, omega0hat)
                + np.kron(omegahat, invOmega0hat)
                - 2 * np.kron(np.eye(u), np.eye(r - u))
            )
            tmp2 = np.kron(etahat.T, gamma0hat)
            covMatrix = np.kron(invsigX, sigma1) + tmp2 @ cholinv(tmp) @ tmp2.T
            asySE = np.sqrt(np.diag(covMatrix))
            ratio = asyFm / asySE
    # update with all the additional stats
    res.update(
        {
            "mu": muhat,
            "beta": betahat,
            "sigma": sigmahat,
            "eta": etahat,
            "omega": omegahat,
            "omega0": omega0hat,
            "loglik": loglik,
            "covMatrix": covMatrix,
            "asySE": asySE,
            "ratio": ratio,
        }
    )
    return res


def fit_xenv_gram(sigY, sigYX, sigX, u, init=None):
    """
    Fit X-envelope model using grammian matrices.
    Called by fit_xenv, which uses full X, Y matrices.
    """
    r = sigY.shape[1]
    p = sigX.shape[1]
    if u > p or u < 0:
        raise ValueError(f"u must be an integer between 0 and r.")
    # check the dim of initial value
    if init is not None:
        if init.shape[0] != r or init.shape[1] != u:
            raise ValueError(f"wrong dimension for init, requires (r={r} x u={u})")

    invsigY = cholinv(sigY)
    betaOLS = sigYX / sigX if sigX.shape[1] == 1 else np.linalg.solve(sigX, sigYX.T)

    U = sigYX.T @ invsigY @ sigYX  # expansion: k x k
    M = sigX - U  # expansion: k x k, dimension of the features

    res = envMU(M, U, u, initial=init)

    gammahat = res["gammahat"]
    gamma0hat = res["gamma0hat"]

    betahat = None
    if u == 0:
        betahat = np.zeros((p, r))
    elif u == p:
        betahat = betaOLS
    else:
        etahat = gammahat.T @ sigYX.T  # reduction to dimension u
        omegahat = gammahat.T @ sigX @ gammahat
        omega0hat = gamma0hat.T @ sigX @ gamma0hat
        invOmegahat = cholinv(omegahat)
        betahat = gammahat @ invOmegahat @ etahat  # expansion from dimension u -> k

    # additional data to be returned
    res.update(
        {
            "u": u,
            "r": r,
            "p": p,
            "U": U,
            "M": M,
            "betaols": betaOLS,  # cannonical OLS solution
            "beta": betahat,  # envelope solution
        }
    )
    return res


def fit_xenv(X, Y, u, asy=True, init=None):
    """
    Fits envelope model of X, given envelope subspace dimension u.
    Returns the envelope modified beta, along with other measures
    involved in solving for projection matrix. X is usually
    multi-features matrix, and Y is a single vector.

    Args:
        X : (n x p)
        Y : (n x 1)
        u : scalar integer between [1, p]

    Returns dict with keys:
        gamma       : projection matrix for material info
        gamma0      : projection matrix for immaterial info
        mu:         : intercept
        beta:       : envelope modified OLS beta
        sigmaX      : modified total variance
        eta         : projected x'y cov in reduced u dimension
        omega:      : projected x'x cov in reduced u dimension (material)
        omega0:     : projected x'x cov in reduced u dimension (immaterial)
        sigmaYcX    : (edit) internal
        loglik      : log likelihood value
        n           : number of samples
        covMatrix   : final covariance
        asySE       : asymptotic SE (standard error) for envelope solution
        ratio       : asymptotic SE of OLS / asymptotic SE of envelope for each beta
        niter       : number of iterations taken in envMU opt
        opt_elapsed : time taken to solve in envMU opt
    """
    n, r = Y.shape
    p = X.shape[1] if X.ndim > 1 else 1
    if n != X.shape[0]:
        raise ValueError(f"X and Y should have the same number of observations.")
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    # NOTE: check for duplicate columns

    nadj = (n - 1) / n
    sigY = cov(Y) * nadj
    sigYX = cov(Y, X) * nadj
    sigX = cov(X) * nadj
    yev, yE = np.linalg.eig(sigY)

    res = fit_xenv_gram(sigY, sigYX, sigX, u, init=init)

    U = res["U"]
    M = res["M"]
    betaOLS = res["betaols"]
    gammahat = res["gammahat"]
    gamma0hat = res["gamma0hat"]
    objfun = res["objfun"]
    covMatrix = None
    asySE = None
    ratio = None

    if u == 0:
        etahat = None
        omegahat = None
        omega0hat = sigX
        muhat = Y.mean(axis=0)
        betahat = np.zeros((p, r))
        sigmaXhat = sigX
        sigmaYcXhat = sigY
        loglik = -n * (p + r) / 2 * (np.log(2 * np.pi) + 1) - n / 2 * (
            objfun + np.log(yev).sum()
        )
        if asy:
            ratio = np.ones((p, r))
    elif u == r:
        invsigX = cholinv(sigX)
        etahat = betaOLS
        omegahat = sigX
        omega0hat = None
        muhat = Y.mean(axis=0) - X.mean(axis=0) @ betaOLS
        betahat = betaOLS
        sigmaXhat = M + U
        sigmaYcXhat = sigY - sigYX @ betaOLS
        loglik = -n * (r + p) / 2 * (np.log(2 * np.pi) + 1) - n / 2 * (
            objfun + np.log(yev).sum()
        )
        if asy:
            covMatrix = np.kron(sigmaYcXhat, invsigX)
            asySE = np.sqrt(np.diag(covMatrix))
            ratio = np.ones((p, r))
    else:
        invsigX = cholinv(sigX)
        etahat = gammahat.T @ sigYX.T  # reduction to dimension u
        omegahat = gammahat.T @ sigX @ gammahat
        omega0hat = gamma0hat.T @ sigX @ gamma0hat
        invOmegahat = cholinv(omegahat)
        betahat = gammahat @ invOmegahat @ etahat  # expansion from dimension u -> k
        muhat = Y.mean(axis=0) - X.mean(axis=0) @ betahat
        sigmaXhat = (
            gammahat @ omegahat @ gammahat.T + gamma0hat @ omega0hat @ gamma0hat.T
        )
        pGamma = gammahat @ gammahat.T
        sigmaYcXhat = sigY - sigYX @ pGamma @ cholinv(sigmaXhat) @ pGamma @ sigYX.T

        loglik = -n * (r + p) / 2 * (np.log(2 * np.pi) + 1) - n / 2 * (
            objfun + np.log(yev).sum()
        )
        if asy:
            covMatrix = np.kron(sigmaYcXhat, invsigX)
            asyFm = np.sqrt(np.diag(covMatrix))
            invSigmaYcXhat = cholinv(sigmaYcXhat)
            invOmega0hat = cholinv(omega0hat)
            tmp = (
                np.kron(etahat @ invSigmaYcXhat @ etahat.T, omega0hat)
                + np.kron(invOmegahat, omega0hat)
                + np.kron(omegahat, invOmega0hat)
                - 2 * np.kron(np.eye(u), np.eye(p - u))
            )
            tmp2 = np.kron(etahat.T, gamma0hat)
            covMatrix = (
                np.kron(sigmaYcXhat, gammahat @ invOmegahat @ gammahat.T)
                + tmp2 @ cholinv(tmp) @ tmp2.T
            )
            asySE = np.sqrt(np.diag(covMatrix))
            ratio = asyFm / asySE
    # return all the stats
    res.update(
        {
            "mu": muhat,
            "sigmaX": sigmaXhat,
            "eta": etahat,
            "omega": omegahat,
            "omega0": omega0hat,
            "sigmaYcX": sigmaYcXhat,
            "loglik": loglik,
            "n": n,
            "covMatrix": covMatrix,
            "asySE": asySE,
            "ratio": ratio,
        }
    )
    return res


class YEnvlp(object):
    """
    Multi-Y envelope fitter compatible with Regression suite of classes.
    Since this is a python object, we keep the members simple such that
    it is pickle-able by python.
    """

    def __init__(self, l2_lambda=0.001, **kw):
        self.result = {
            "name": "yenvregression",
            "l2_lambda": l2_lambda,  # NOTE: currently not used
            "predict_index": kw.get("predict_index", -1),
            "apply_sigma": kw.get("apply_sigma") or False,
        }

    def calc_gram(self, x, y, **kwargs):
        # use the gram func from pylinear
        import pylinear

        xtx, xty, yty, n = None, None, None, x.shape[0]
        if kwargs.get("weight", None) is not None and len(kwargs["weight"]) > 0:
            w = kwargs["weight"]
            if w.shape[0] != x.shape[0]:
                raise ValueError("{type(self).__name__}: weight shape mismatch")
            xtx, xty, yty = pylinear.GramUtil.dotw(x, y, kwargs["weight"], True)
        else:
            xtx, xty, yty = pylinear.GramUtil.dot(x, y, True)
        # note that we explicitly return the size of the original matrix
        return xtx, xty, yty, n

    def fit(self, x, y, w=np.array([])):
        # in the presence of weights, it will apply to gram matrices
        xtx, xty, yty, nrows = self.calc_gram(x, y, weight=w)
        # calls fit gram
        self.fit_gram(xtx, xty, yty, n=nrows)

    def fit_gram(self, xtx: np.ndarray, xty: np.ndarray, yty: np.ndarray, nrows: int):
        if xtx.ndim != 2 or xtx.shape[0] != xtx.shape[1]:
            raise ValueError(f"{type(self).__name__}: invalid xtx dims {xtx.shape=}")
        if xty.shape[0] != xtx.shape[0]:
            raise ValueError(f"{type(self).__name__}: invalid xty dims {xty.shape=}")
        meanY = xty[[0], :].T / nrows
        meanX = xtx[1:, [0]] / nrows
        sigY = yty / nrows - meanY @ meanY.T
        sigX = xtx[1:, 1:] / nrows - meanX @ meanX.T
        sigXY = xty[1:, :] / nrows - meanX @ meanY.T
        sigYX = sigXY.T
        r = sigY.shape[1]  # size of response (Y targets)
        p = sigX.shape[1]  # size of predictors (X features)
        bic_yenv = []
        loglik_yenv = []
        reslist = []
        # could leverage envMU directly to parallelize here
        for i in range(0, (r + 1)):
            # does the actual work with envMU
            res = fit_yenv_gram(sigY, sigYX, sigX, i)
            loglik = (
                -nrows * r / 2 * (np.log(2 * np.pi) + 1) - nrows / 2 * res["objfun"]
            )
            bic = -2 * loglik + np.log(nrows) * (r + r * (r + 1) / 2 + p * i)
            loglik_yenv.append(loglik)
            bic_yenv.append(bic)
            reslist.append(res)
        # find the index of the min bic value
        uidx = np.argmin(np.array(bic_yenv))
        bestres = reslist[uidx]
        # add more stats
        bestres.update(
            {
                "xsum": meanX * nrows,
                "ysum": meanY * nrows,
                "nrows": nrows,
                "loglik": np.array(loglik_yenv),
                "bic": np.array(bic_yenv),
                "best_u": uidx,
                "mu_env": meanY - res["beta"] @ meanX,  # intercept
            }
        )
        # now save the results into fitter member for serialization
        self.result.update(bestres.copy())

    def predict(self, x: np.ndarray, include_intercept: bool = True):
        """
        Currently include_intercept is ignored.
        """
        if "mu_env" not in self.result:
            raise KeyError(f"{type(self).__name__}: 'mu_env' not found in result")

        mu_env = self.result["mu_env"]
        beta = self.coefs()
        yhmat = x @ beta.T + mu_env.T
        # ensemble (average) the predictions
        yh = yhmat.mean(axis=1)
        # but pick the specific yh if we specified it
        idx = self.result.get("predict_index", -1)
        if idx >= 0 and idx < yhmat.shape[1]:
            yh = yhmat[:, self.predict_index]
        return yh

    def coefs(self):
        if "beta" not in self.result:
            raise KeyError(f"{type(self).__name__}: 'beta' coef not found in result")
        return self.result["beta"]

    @property
    def coef_(self):
        return self.coefs()

    def coefsbase(self):
        if "betaols" not in self.result:
            raise KeyError(f"{type(self).__name__}: 'betaols' coef not found in result")
        return self.result["betaols"]

    def get_nrows(self):
        if "nrows" not in self.result:
            raise KeyError(f"{type(self).__name__}: 'nrows' not found in result")
        return self.result["nrows"]

    def get_xsum(self):
        if "xsum" not in self.result:
            raise KeyError(f"{type(self).__name__}: 'xsum' not found in result")
        return self.result["xsum"]

    def get_ysum(self):
        if "ysum" not in self.result:
            raise KeyError(f"{type(self).__name__}: 'ysum' not found in result")
        return self.result["ysum"]

    def get_result(self):
        return self.result

    def get_param(self, k):
        return self.result.get(k)


# for convenience, and for familiarity with R lib
env = fit_env = fit_yenv
