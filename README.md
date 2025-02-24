# Envelope Models

The envelope model is a method for efficient estimation in multivariate linear regression.
This is an implementation that largely follows the algorithm already available in R
as `Renvlp` package.

While we are largely interested in "predictor" envelopes, the implementation
is generalizable to "response" envelopes as well.

# Installation

You will need to do this on `linux` machine, also have `cmake` and
`build-essentials` (compiler, toolchain) already installed.
In most cases, it should be sufficient to `git clone` this repo, 
and run inside the `envelope` directory.


```
cmake -S . -B build
cmake --build build -j
cmake --build build --target test
```


# C++ envelope library

The library implements optimization routine using fast L-BFGS routine that
exploits SSE2 optimized vector routines. This requires modern CPU's capable
of handling AVX/SSE2 instructions in somewhat homogeneous computing environment.

* [la_util.hpp](./cpp/include/envelope/la_util.hpp) :
    Implements linear algebra functions that are required for envelope optimization.

* [envopt.hpp](./cpp/include/envelope/envopt.hpp) :
    Core of the envelope model where optimization solves for `gammahat`, `gamma0hat`,
    given inputs `M` and `U` matrices.

* [regression_env.hpp](./cpp/include/envelope/regression_env.hpp)
    Implements multi-X predictor envelope `RidgeEnvlp` fitter, compatible with
    `fitter::linear` fitters. This object can be used within `blender.py` as
    part of available fitters for prediction problem.


# pyenvlp

Fitting the envelope model involves an objective that needs to be solved using optimization
techniques. This module separates the implementation into 3 logical components:

* [pyenvlp.opt](./python/src/pyenvlp/opt.py) :
    Implements optimization routine to solve for gammahat, gamma0hat in python.
    Uses the cpp optimization library if exists.

* [pyenvlp.fit](./python/src/pyenvlp/fit.py) :
    function that accepts X, Y numpy arrays, constructs the necessary
    input into the optimization function, and creates evaluation stats
    using the optimized results.

* [pyenvlp.lau](./python/src/pyenvlp/lau.py) :
    linear algebra utility functions


# `envMU` Optimization Notes

## Inputs

* u  : dimension of the envelope subspace
* p  : dimension of predictors (columns of X)
* r  : dimension of response (columns of Y)

For "response" envelope where Y contains multiple response.

* U  : `betaOLS' * (x'y)`, where dims (t x k) * (k x t) => (t x t).
       Evaluates to: `(x'y)' (x'x)^-1 (x'y)`.
       This is expansion of beta by each cov of x'y, looking at
       permutation of each beta coef by each target dimension. Expresses
       all possible covariance of the betas in the target space.
* M  : `y'y - U`, where dims are also t x t, this is covariance mat
       of the residuals on Y.
* MU : M + U, marginal sample cov of Y. MU = cov(Y), dim (t x t)

For case where single-Y and multi-X (our usecase):

* U  : `(x'y) inv(y'y) * (x'y)'`, dim (k x k), span of cov(x,y)
* M  : `x'x - U`, dim (k x k)
* MU : `x'x`, dim (k x k), effectively cov of X. MU = cov(X)


## Initial value calculation

* initial value has the dimension of `(p * u)`, where `p` is the number of
  targets or features columns, and `u` is the dimension of the envelope subspace.
* eigen is used to approximate the inverse
* 4 distinct initial values are tried, where each eigen vectors of M, MU
  are combined with "midmatrix" to figure out which eigen vector
  has the largest value. This largest eigen vector is then used as
  initial value, which has the same dimension as the number of targets


## Considerations

* `u` denotes the dimension of the envelope subspace, and associated with
  dimension of the likely material information contained in the input data.
* the optimization routine has a special case for when `u=1`, `u=r-1`, where solution
  is reached faster solved without expensive coordinate descent.
* case where `u > 1 & u < r - 1` requires coordinate descent in the optimization loop.
* L-BFGS algorithm has fast C impl which improves the runtime of optimization drastically.
* the 4 initial value calculation can be parallelized, but performance gain would be small.
* use of gaussian elimination, use of QR decomp, use of cholesky-inverse.


# References

* Renvlp: https://cran.r-project.org/web/packages/Renvlp/index.html
* paper 1: https://people.clas.ufl.edu/zhihuasu/files/Biometrika-2016-Su-579-93.pdf
* paper 2: https://people.clas.ufl.edu/zhihuasu/files/review-6.pdf
* book: https://www.amazon.com/Introduction-Envelopes-Estimation-Multivariate-Probability/dp/1119422930
