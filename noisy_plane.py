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
    return samples

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
#     return np.logaddexp.reduce(loglike, axis=0)

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

#     plt.clf()
#     plt.plot(samples[0, 1, :], samples[1, 1, :], "r.", markersize=2)
#     plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0)
