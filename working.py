import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as spo
import emcee
import triangle
from plotting_working import load_dat
import pretty5
from subgiants import MS_poly
from working_likes import lnlike

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
pl.rcParams.update(plotpar)

Tk = 6250

def log_period_model(par, log_age, temp):
    return par[0] + par[1] * log_age + par[2] * np.log10(Tk - temp)

def lnprior(m):
    if -10. < m[0] < 10. and .3 < m[1] < .8 and 0. < m[2] < 1. \
            and 0 < m[3] < np.log10(30.) and 0 < m[4] < np.log10(100.)\
            and 0 < m[5] < np.log10(30.) and 0 < m[6] < np.log10(100.)\
            and 0 < m[7] < np.log10(30.) and 0 < m[8] < np.log10(100.)\
            and 0. < m[9] < 1.:
        return 0.0
    return -np.inf

def lnprob(m, log_age_samp, temp_samp, log_period_samp, logg_samp, \
        temp_obs, temp_err, log_period_obs, log_period_err, logg_obs, \
        logg_err, coeffs, Tk):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m, log_age_samp, temp_samp, \
            log_period_samp, logg_samp, temp_obs, temp_err, log_period_obs, \
            log_period_err, logg_obs, logg_err, coeffs, Tk)

def MCMC(fname):
    # log(a), n, beta, Y, V, Z, U
    par_true = [np.log10(0.7725), 0.5189, .2, np.log10(5.), np.log10(10.), \
            np.log10(10.), np.log10(10.), np.log10(1.5), np.log10(5.), .5]

    # load real data
    log_period_obs, temp_obs, log_age_obs, log_period_err, temp_err, log_age_err, \
            log_age_errp, log_age_errm, \
            logg_obs, logg_err, logg_errp, logg_errm, age, age_errp, \
            age_errm, period, period_err = load_dat()

    # plot period vs age
    pl.clf()
    pl.errorbar(10**log_age_obs, 10**log_period_obs, xerr=10**log_age_err, \
            yerr=10**log_period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')
    log_age_plot = np.linspace(0, max(log_age_obs))
    pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 6000.), 'r-')
    pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 5500.), 'm-')
    pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 5000.), 'b-')
    pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 4000.), 'c-')
    pl.xlabel('$\mathrm{Age~(Gyr)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    pl.savefig("init%s" %fname)

    # plot period vs teff
    pl.clf()
    pl.errorbar(temp_obs, 10**log_period_obs, xerr=temp_err, yerr=10**log_period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')
    temp_plot = np.linspace(min(temp_obs), max(temp_obs))
    pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(1.), temp_plot), 'r-')
    pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(2.), temp_plot), 'm-')
    pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(5.), temp_plot), 'b-')
    pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(10.), temp_plot), 'c-')
    pl.xlim(pl.gca().get_xlim()[::-1])
    pl.xlabel('$\mathrm{T_{eff}~(K)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    pl.savefig("init_teff%s" %fname)

    # Now generate samples
    nsamp = 100
    log_age_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_age_obs, log_age_err)])
    temp_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(temp_obs, temp_err)])
    logg_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(logg_obs, logg_err)])
    log_period_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_period_obs, log_period_err)])

    # calculate ms turnoff coeffs
    coeffs = MS_poly()
    turnoff = np.polyval(coeffs, temp_obs)-.1

    print 'initial likelihood = ', lnlike(par_true, log_age_samp, temp_samp, \
            log_period_samp, logg_samp, temp_obs, temp_err, log_period_obs, log_period_err, \
            logg_obs, logg_err, coeffs, Tk)

    nwalkers, ndim = 32, len(par_true)
    p0 = [par_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (log_age_samp, temp_samp, log_period_samp, logg_samp, temp_obs, \
            temp_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, Tk)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)

    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, 500)
    sampler.reset()
    print("Production run")
#     sampler.run_mcmc(p0, 10000)
    nstep = 10000
    nruns = 500.

    for j in range(int(nstep/nruns)):

        print 'run', j
        p0, lp, state = sampler.run_mcmc(p0, nruns)

        print("Plotting traces")
        pl.figure()
        for i in range(ndim):
            pl.clf()
            pl.axhline(par_true[i], color = "r")
            pl.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3)
            pl.savefig("%s%s.png" %(i, fname))

        flat = sampler.chain[:, 50:, :].reshape((-1, ndim))
        mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                          zip(*np.percentile(flat, [16, 50, 84], axis=0)))
        mcmc_result = np.array(mcmc_result)[:, 0]
        print 'mcmc_result = ', mcmc_result

        print("Making triangle plot")
        fig_labels = ["$a$", "$n$", "$b$", "$Y$", "$V$", "$Z$", "$U$", "$X$", "$W$", "$P$"]
        fig = triangle.corner(sampler.flatchain, truths=mcmc_result, labels=fig_labels[:len(par_true)])
        fig.savefig("triangle%s.png" %fname)

    # Flatten chain
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    # Find values
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(samples, [16, 50, 84], axis=0)))

    print 'initial values', par_true
    mcmc_result = np.array(mcmc_result)[:, 0]
    print 'mcmc result', mcmc_result

    print("Making triangle plot")
    fig_labels = ["$log(a)$", "$n$", "$b$", "$Y$", "$V$", "$Z$", "$U$", "$X$", "$W$", "$P$"]
    fig = triangle.corner(sampler.flatchain, truths=mcmc_result, labels=fig_labels[:len(par_true)])
    fig.savefig("triangle%s.png" %fname)

    mcmc_result = [mcmc_result[0], mcmc_result[1], mcmc_result[2], 6250, \
            mcmc_result[3], mcmc_result[4]]

    # plot result
    pl.clf()
    pl.errorbar(10**log_age_obs, 10**log_period_obs, xerr=log_age_err, yerr=10**log_period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')
    log_age_plot = np.linspace(0, max(log_age_obs))
    pl.plot(10**log_age_plot, 10**log_period_model(mcmc_result, log_age_plot, 6000.), 'r-')
    pl.plot(10**log_age_plot, 10**log_period_model(mcmc_result, log_age_plot, 4000.), 'b-')
    pl.plot(10**log_age_plot, 10**log_period_model(mcmc_result, log_age_plot, 5000.), 'm-')
    pl.plot(10**log_age_plot, 10**log_period_model(mcmc_result, log_age_plot, 4500.), 'c-')
    pl.xlabel('$\mathrm{Age~(Gyr)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    pl.savefig("result%s" %fname)

    # plot period vs teff
    pl.clf()
    pl.errorbar(temp_obs, 10**log_period_obs, xerr=temp_err, yerr=10**log_period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')
    temp_plot = np.linspace(min(temp_obs), max(temp_obs))
    pl.plot(temp_plot, 10**log_period_model(mcmc_result, np.log10(1.), temp_plot), color=ocols[0],\
             linestyle='-')
    pl.plot(temp_plot, 10**log_period_model(mcmc_result, np.log10(2.), temp_plot), color=ocols[1],\
             linestyle='-')
    pl.plot(temp_plot, 10**log_period_model(mcmc_result, np.log10(5.), temp_plot), color=ocols[2],\
             linestyle='-')
    pl.plot(temp_plot, 10**log_period_model(mcmc_result, np.log10(10.), temp_plot), color=ocols[3],\
             linestyle='-')
    pl.xlim(pl.gca().get_xlim()[::-1])
    pl.xlabel('$\mathrm{T_{eff}~(K)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    pl.savefig("result_teff%s" %fname)

if __name__ == "__main__":

    MCMC('work')
