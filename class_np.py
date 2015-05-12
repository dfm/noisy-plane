import numpy as np
import matplotlib.pyplot as plt
from model import model, model1

class noisy_plane(object):

    def __init__(self, obs, u):
        self.obs = obs
        self.u = u

    def generate_samples(self, N):
        '''
        obs is a 2darray of observations. shape = (ndims, nobs)
        u in an ndarray of the uncertainties associated with the observations
        returns a 3d array of samples. shape = (nobs, ndims, nsamples)
        '''
        ndims, nobs = np.shape(self.obs)
        samples = np.zeros((ndims, nobs, N))
        for i in range(ndims):
            samples[i, :, :] = np.vstack([x0+xe*np.random.randn(N) for x0, xe in \
                    zip(self.obs[i, :], self.u[i, :])])
        return samples

    # n-D non-hierarchical log-likelihood
    def lnlike(self, pars, samples):
        '''
        Generic likelihood function for importance sampling with any number of
        dimensions.
        In samples, obs and u, the dependent variable should be 1st
        '''
        ndims, nobs, nsamp = samples.shape
        zpred = model(pars, samples)
        zobs = self.obs[1, :]
        zerr = self.u[1, :]
        ll = np.zeros((nobs, nsamp*nobs))
        for i in range(nobs):
            ll[i, :] = -.5*((zobs[i] - zpred)/zerr[i])**2
        loglike = np.sum(np.logaddexp.reduce(ll, axis=1))
        return loglike

    # n-D, hierarchical log-likelihood
    def lnlikeH(self, pars, samples):
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
        zobs = self.obs[1, :]
        zerr = self.u[1, :]
        ll = np.zeros((nobs, nsamp*nobs))
        for i in range(nobs):
            inv_sigma2 = 1.0/(zerr[i] + pars[2])**2
            ll[i, :] = -.5*((zobs[i] - zpred)**2*inv_sigma2) + np.log(inv_sigma2)
        loglike = np.sum(np.logaddexp.reduce(ll, axis=1))
        return loglike

    # 1d, hierarchical log-Likelihood
    def orig_lnlike(self, theta, x, y, yerr):
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
