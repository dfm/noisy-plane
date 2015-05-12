import numpy as np
import matplotlib.pyplot as plt
from model import model, model1

def generate_samples(obs, u, N):
    '''
    obs is a 2darray of observations. shape = (ndims, nobs)
    u in an ndarray of the uncertainties associated with the observations
    returns a 3d array of samples. shape = (nobs, ndims, nsamples)
    '''
    ndims, nobs = np.shape(obs)
    samples = np.zeros((ndims, nobs, N))
    for i in range(ndims):
        samples[i, :, :] = np.vstack([x0+xe*np.random.randn(N) for x0, xe in \
                zip(obs[i, :], u[i, :])])
#     plt.clf()
#     x, y = obs[0, :], obs[1, :]
#     xerr, yerr = u[0, :], u[1, :]
#     plt.plot(samples[0, :, :], samples[1, :, :], "r.", markersize=2, alpha=.3)
#     plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0, ecolor=".7")
#     plt.show()
#     assert 0
    return samples

# n-D non-hierarchical log-likelihood
def lnlike(pars, samples, obs, u):
    '''
    Generic likelihood function for importance sampling with any number of
    dimensions.
    In samples, obs and u, the dependent variable should be 1st
    '''
    ndims, nobs, nsamp = samples.shape
    zpred = model(pars, samples)
    zobs = obs[1, :]
    zerr = u[1, :]
    ll = np.zeros((nobs, nsamp*nobs))
    for i in range(nobs):
        ll[i, :] = -.5*((zobs[i] - zpred)/zerr[i])**2
    loglike = np.sum(np.logaddexp.reduce(ll, axis=1))
    return loglike

# n-D, hierarchical log-likelihood
def lnlikeH(pars, samples, obs, u):
    '''
    Generic likelihood function for importance sampling with any number of
    dimensions.
    Now with added jitter parameter (hierarchical)
    obs should be a 2d array of observations. shape = (ndims, nobs)
    u should be a 2d array of uncertainties. shape = (ndims, nobs)
    samples is a 3d array of samples. shape = (ndims, nobs, nsamp)
    '''
    ndims, nobs, nsamp = samples.shape
    zpred = model(pars, samples)
    zobs = obs[1, :]
    zerr = u[1, :]
    ll = np.zeros((nobs, nsamp*nobs))
    for i in range(nobs):
        inv_sigma2 = 1.0/(zerr[i]**2 + pars[2]**2)
        ll[i, :] = -.5*((zobs[i] - zpred)**2*inv_sigma2) + np.log(inv_sigma2)
    loglike = np.sum(np.logaddexp.reduce(ll, axis=1))
    return loglike

# n-D, hierarchical, mixture-model log-likelihood
def lnlikeHM(pars, samples, obs, u):
    '''
    Generic likelihood function for importance sampling with any number of
    dimensions.
    Now with added jitter parameter (hierarchical) and a mixture model.
    obs should be a 2d array of observations. shape = (ndims, nobs)
    u should be a 2d array of uncertainties. shape = (ndims, nobs)
    samples is a 3d array of samples. shape = (ndims, nobs, nsamp)
    Y and V are the mean and variance of the Gaussian.
    P is the probability that a data point is drawn from the Gaussian
    '''
    ndims, nobs, nsamp = samples.shape
    zpred = model(pars, samples)
    zobs = obs[1, :]
    zerr = u[1, :]
    Y, V, P = pars[3:]
    ll1 = np.zeros((nobs, nsamp*nobs))
    ll2 = np.zeros((nobs, nsamp*nobs))
    ll = np.zeros((nobs, nsamp*nobs))
    for i in range(nobs):
        inv_sigma21 = 1.0/(zerr[i]**2 + pars[2]**2)
        inv_sigma22 = 1.0/(zerr[i]**2 + V)
        ll1[i, :] = -.5*((zobs[i] - zpred)**2*inv_sigma21) + np.log(inv_sigma21)
        ll2[i, :] = -.5*((zobs[i] - Y)**2*inv_sigma22) + np.log(inv_sigma22)
        ll[i, :] = np.logaddexp(np.log(1-P) + ll1[i, :], np.log(P) + ll2[i, :])
    loglike = np.sum(np.logaddexp.reduce(ll, axis=1))
    return loglike

# 1d, hierarchical log-Likelihood
def orig_lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

if __name__ == "__main__":
    pars = [.5, 10]
    # make fake data
    nobs = 20
    x = np.random.uniform(0, 10, nobs)
    xerr = np.ones_like(x) * .2
    x += np.random.randn(nobs) * .5
    y = pars[0] * x + pars[1]
    yerr = np.ones_like(y) * .5
    y += np.random.randn(nobs) * .5

    obs = np.vstack((y, x))  # dependent variable 1st
    u = np.vstack((yerr, xerr))
    nsamp = 50
    samples = generate_samples(obs, u, nsamp)
    print lnlike(pars, samples, obs, u)

    plt.clf()
    plt.plot(samples[0, 1, :], samples[1, 1, :], "r.", markersize=2)
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0)
    plt.show()
