import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as spo
import emcee
import triangle
from plotting import load_dat
import pretty5
from subgiants import MS_poly
from lnlikes import lnlike, neglnlike
from scipy.optimize import fmin

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
pl.rcParams.update(plotpar)

def lnprior(m):
    if -10. < m[0] < 10. and .3 < m[1] < .8 and 0. < m[2] < 1. \
            and 0 < m[3] < np.log10(30.) and 0 < m[4] < np.log10(100.)\
            and 0 < m[5] < np.log10(30.) and 0 < m[6] < np.log10(100.):
        return -.5*((m[1]-.5189)/.2)**2 -.5*((m[2]-.2)/.2)**2
#         return 0.0
    return -np.inf

def lnprob(m, log_age_samp, temp_samp, log_period_samp, \
        temp_obs, temp_err, log_period_obs, log_period_err, coeffs):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m, log_age_samp, temp_samp, \
            log_period_samp, logg_samp, temp_obs, temp_err, log_period_obs, \
            log_period_err, logg_obs, logg_err, coeffs, Tk)
def MCMC(par, log_age_samp, temp_samp, log_period_samp, logg_samp, temp_obs, \
        temp_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, Tk):

    nwalkers, ndim = 32, len(par_true)
    p0 = [par_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (log_age_samp, temp_samp, log_period_samp, temp_obs, temp_err, log_period_obs, \
            log_period_err, coeffs)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)

    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, 500)
    sampler.reset()
    print("Production run")
    sampler.run_mcmc(p0, 2000)

#     print("Plotting traces")
#     pl.figure()
#     for i in range(ndim):
#         pl.clf()
#         pl.axhline(par_true[i], color = "r")
#         pl.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3)
#         pl.savefig("{0}.png".format(i))

    # Flatten chain
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    # Find values
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(samples, [16, 50, 84], axis=0)))

    print 'initial values', par_true
    mcmc_result = np.array(mcmc_result)[:, 0]
    print 'mcmc result', mcmc_result

#     print("Making triangle plot")
#     fig_labels = ["$log(a)$", "$n$", "$b$", "$Y$", "$V$", "$Z$", "$U$"]
#     fig = triangle.corner(sampler.flatchain, truths=mcmc_result, labels=fig_labels[:len(par_true)])
#     fig.savefig("triangle.png")

    return mcmc_result

def maxlike(par, log_age_samp, temp_samp, log_period_samp, logg_samp, temp_obs, \
        temp_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, Tk):

    args = (log_age_samp, temp_samp, log_period_samp, logg_samp, temp_obs, temp_err, \
            log_period_obs, log_period_err, logg_obs, logg_err, coeffs, Tk)

    print 'initial likelihood = ', lnlike(par, log_age_samp, temp_samp, log_period_samp, logg_samp, temp_obs, \
            temp_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, Tk)

#     result = fmin(neglnlike, par, args=args)
    result = MCMC(par, log_age_samp, temp_samp, log_period_samp, logg_samp, temp_obs, \
        temp_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, Tk)

    print 'result = ', result
    likelihood = lnlike(result, log_age_samp, temp_samp, log_period_samp, logg_samp, temp_obs, \
            temp_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, Tk)

    print 'final likelihood = ', likelihood
    return likelihood

if __name__ == "__main__":
    # log(a), n, beta, Y, V, Z, U
    par_true = [np.log10(0.7725), 0.5189, .2, np.log10(5.), np.log10(10.), \
            np.log10(10.), np.log10(10.)]

    # load real data
    log_period_obs, temp_obs, log_age_obs, log_period_err, temp_err, log_age_err, \
            logg_obs, logg_err = load_dat()

    # Now generate samples
    nsamp = 100
    log_age_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_age_obs, log_age_err)])
    temp_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(temp_obs, temp_err)])
    logg_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(logg_obs, logg_err)])
    log_period_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_period_obs, log_period_err)])
    # FIXME: asymmetric errorbars for age and logg

    # calculate ms turnoff coeffs
    coeffs = MS_poly()

    # Grid over Tk
    K_temps = np.arange(6100, 6700, 50)
#     K_temps = np.ones(5)*6250
    L = np.empty_like(K_temps)

    for i, Tk in enumerate(K_temps):
        print "TEMP = ", Tk
        L[i] = maxlike(par_true, log_age_samp, temp_samp, log_period_samp, logg_samp, temp_obs, \
                temp_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, Tk)

    pl.clf()
    pl.plot(K_temps, L, 'k-')
    pl.xlabel('Tk')
    pl.ylabel('Likelihood')
    pl.savefig('Tk_likelihood')
