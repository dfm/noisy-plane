import numpy as np
import matplotlib.pyplot as plt
from noisy_plane import generate_samples, lnlike, lnlikeH
from model import model1, model
import emcee
import triangle

def lnprior(pars):
    return 0.

def lnprob(pars, samples, obs, u):
    return lnlikeH(pars, samples, obs, u) + lnprior(pars)

if __name__ == "__main__":
    pars = [.5, 10, .1]

    # make fake data
    nobs = 20
    x = np.random.uniform(0, 10, nobs)
    xerr = np.ones_like(x) * .2
    x += np.random.randn(nobs) * .5
    y = pars[0] * x + pars[1]
    yerr = np.ones_like(y) * .5
    y += np.random.randn(nobs) * .5

    obs = np.vstack((x, y))
    u = np.vstack((xerr, yerr))
    nsamp = 3
    s = generate_samples(obs, u, nsamp)

    pars_init = [.3, 12, .1]

    ndim, nwalkers = len(pars_init), 32
    pos = [pars_init + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(s, obs, u))

    print "burning in..."
    pos, _, _, = sampler.run_mcmc(pos, 500)
    sampler.reset()
    print "production run..."
    sampler.run_mcmc(pos, 1000)
    samp = sampler.chain[:, 50:, :].reshape((-1, ndim))
    m, c = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
               zip(*np.percentile(samp, [16, 50, 84], axis=0)))
    pars = [m[0], c[0]]

    plt.clf()
    plt.plot(s[0, :, :], s[1, :, :], "r.", markersize=2)
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0)
    print pars_init, pars
    plt.plot(x, model1(pars_init, x))
    plt.plot(x, model1(pars, x))
    plt.savefig("mcmc")

    fig = triangle.corner(samp, truths=pars_init)
    fig.savefig("triangle.png")
