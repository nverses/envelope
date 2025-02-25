#!/usr/bin/env python3

import os, sys
import time
import pandas as pd
import numpy as np
import getpass
import pyenvlp
from pyenvlp.lau import cov, geindex, cholinv

np.set_printoptions(linewidth=200, suppress=True)


def find_dataroot():
    if "TEST_DATA_DIR" in os.environ:
        return os.environ["TEST_DATA_DIR"]
    pwd = os.path.dirname(os.path.realpath(__file__))
    for candidate_path in [
        f"{pwd}/../../envelope/data",
        f"{pwd}/../../../envelope/data",
    ]:
        if os.path.exists(candidate_path):
            return candidate_path
    return None


DATAROOT = find_dataroot()


def test_utils():
    testfile = f"{DATAROOT}/wheatprotein.csv"
    df = pd.read_csv(testfile)
    y = df.iloc[:, :-1].values
    ycov = cov(y)
    idx = geindex(ycov)
    print(idx)
    print(np.linalg.cholesky(ycov))
    ycovinv = cholinv(ycov)
    xx = np.linalg.solve(ycov, np.eye(ycov.shape[0]))


def _get_wheat_data():
    testdata = f"{DATAROOT}/wheatprotein.csv"
    df = pd.read_csv(testdata)
    X = df.iloc[:, -1].values
    Y = df.iloc[:, :-1].values
    n = Y.shape[0]
    r = Y.shape[1]
    nadj = (n - 1) / n
    sigY = np.cov(Y.T) * nadj
    sigYX = np.cov(Y.T, X)[-1][:-1] * nadj
    sigX = np.cov(X) * nadj
    betaOLS = sigYX / sigX if sigX.ndim == 0 else np.linalg.solve(sigX, sigYX)
    U = betaOLS[:, np.newaxis] @ sigYX[:, np.newaxis].T
    M = sigY - U
    print(betaOLS)
    return {
        "X": X,
        "Y": Y,
        "n": n,
        "r": r,
        "sigY": sigY,
        "sigYX": sigYX,
        "sigX": sigX,
        "betaOLS": betaOLS,
        "M": M,
        "U": U,
    }


def test_opt():
    # now ready for checking against envMU
    d = _get_wheat_data()
    sigY, sigYX, sigX = d["sigY"], d["sigYX"], d["sigX"]
    betaOLS, M, U = d["betaOLS"], d["M"], d["U"]
    # note this expansion
    print(U)
    print(M)
    res = pyenvlp.envMU(M, U, 1)
    for k, v in res.items():
        print(f"\n{k}\n{v}")

    t0 = time.time()
    res2 = pyenvlp.envMU(M, U, 2)
    res3 = pyenvlp.envMU(M, U, 2)
    for k, v in res2.items():
        print(f"\n{k}\n{v}")
    print(f"envMU_un took: {time.time() - t0:.4f} secs")


def test_fit():
    testdata = f"{DATAROOT}/wheatprotein.csv"
    df = pd.read_csv(testdata)

    # y-env
    X = df.iloc[:, [-1]].values
    Y = df.iloc[:, :-1].values
    res = pyenvlp.fit_yenv(X, Y, 1)
    # y-env gram based
    n = X.shape[0]
    nadj = (n - 1) / n
    xtx = cov(X) * nadj
    ytx = cov(Y, X) * nadj
    yty = cov(Y) * nadj
    res2 = pyenvlp.fit_yenv_gram(yty, ytx, xtx, 1)

    # x-env
    print(res)

    Y = df.iloc[:, [-1]].values
    X = df.iloc[:, :-1].values
    res = pyenvlp.fit_xenv(X, Y, 4)
    print(res)


def test_binding():
    d = _get_wheat_data()
    sigY, sigYX, sigX = d["sigY"], d["sigYX"], d["sigX"]
    betaOLS, M, U = d["betaOLS"], d["M"], d["U"]

    eo = pyenvlp.EnvOpt(M, U)
    eo.solve(2)
    g, g0 = eo.get_gammas()
    ov = eo.get_objvalue()
    print(g)
    print(g0)
    print(ov)


def test_regression_env():
    testdata = f"{DATAROOT}/waterstrider.csv"
    df = pd.read_csv(testdata)
    Y = df.iloc[:, [0]].values
    X = df.iloc[:, 1:].values
    re = pyenvlp.RidgeEnvlp(False)
    re.fit(X, Y)
    coefbase = re.coefsbase()
    coefenv = re.coefs()
    assert coefbase.shape[0] == coefenv.shape[0]

    re.fit_gram(X.T @ X, X.T @ Y, Y.T @ Y, X.shape[0])
    r = X.shape[1]
    u = re.get_u()
    gamma = re.gamma()
    gamma0 = re.gamma0()
    assert gamma.shape[1] == u
    assert gamma0.shape[1] == (r - u)


def check_timing():
    n = 10000
    k = 200
    X = np.random.rand(n, k)
    Y = np.random.rand(n, 1)
    stats = []
    for u in [1, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 199]:
        t0 = time.time()
        res = pyenvlp.fit_xenv(X, Y, u)
        elapsed = time.time() - t0
        stat = {"n": n, "k": k, "u": u, "tot_elapsed": elapsed}
        stat.update({k: res.get(k) for k in ["loglik", "niter", "opt_elapsed"]})
        stats.append(stat)
        print(f"run u={u}: {stat}")
    # save the stats
    sdf = pd.DataFrame(stats)
    print(sdf)
    # exproot = "/dat/pm_nverses/dev/research/experimental/envelope/tmp/"
    # ymdhms = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    # sdf.to_csv(f"{exproot}/envlp_benchmark.{ymdhms}.csv", index=False)


if __name__ == "__main__":
    test_utils()
    test_opt()
    test_fit()
    test_binding()
    test_regression_env()
    # check_timing()
