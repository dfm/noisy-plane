import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as spo
import emcee
import triangle
from plotting import load_dat, log_errorbar
import pretty5
from subgiants import MS_poly
from working_likes import lnlike
from teff_bv import teff2bv

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
pl.rcParams.update(plotpar)

c = .4

def log_period_model(par, log_age, bv):
    return par[0] + par[1] * (log_age+3) + par[2] * np.log10(bv-c)

def lnprior(m):
    if -10. < m[0] < 10. and .3 < m[1] < .8 and 0. < m[2] < 1. \
            and 0 < m[3] < np.log10(30.) and 0 < m[4] < np.log10(100.)\
            and 0 < m[5] < np.log10(30.) and 0 < m[6] < np.log10(100.)\
            and 0 < m[7] < np.log10(30.) and 0 < m[8] < np.log10(100.)\
            and 0. < m[9] < 1.:
        return 0.0
    return -np.inf

def lnprob(m, log_age_samp, bv_samp, log_period_samp, logg_samp, \
        bv_obs, bv_err, log_period_obs, log_period_err, logg_obs, \
        logg_err, coeffs, c):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m, log_age_samp, bv_samp, \
            log_period_samp, logg_samp, bv_obs, bv_err, log_period_obs, \
            log_period_err, logg_obs, logg_err, coeffs, c)

def MCMC(fname):
    # log(a), n, beta, Y, V, Z, U
    par_true = [np.log10(0.7725), 0.5189, .601, np.log10(5.), np.log10(10.), \
            np.log10(10.), np.log10(10.), np.log10(1.5), np.log10(5.), .5]

    # load real data
    log_period_obs, temp_obs, log_age_obs, log_period_err, temp_err, log_age_err, log_age_errp, log_age_errm, \
            logg_obs, logg_err, logg_errp, logg_errm = load_dat()

    # convert temps to bvs
    bv_obs = teff2bv(temp_obs, logg_obs, np.ones_like(temp_obs)*-.2)

    # add praesepe
    data = np.genfromtxt("/Users/angusr/Python/Gyro/data/praesepe.txt").T
    p = 1./data[3]
    p_err = p*(data[4]/data[3])
    bv = data[5]-data[6]
    log_period_obs = np.concatenate((log_period_obs, np.log10(p)))
    log_period_err = np.concatenate((log_period_err, (log_errorbar(p, p_err, p_err))[0]))
    bv_obs = np.concatenate((bv_obs, bv))
    bv_err = np.ones_like(bv_obs)*0.01 # made up for now
    log_age_obs = np.concatenate((log_age_obs, np.ones_like(p)*np.log10(.59)))
    log_age_err = np.concatenate((log_age_err, np.ones_like(p)*\
            (log_errorbar(.59, .01, .01))[0])) # made up age errs

    # add hyades
    data = np.genfromtxt("/Users/angusr/Python/Gyro/data/hyades.txt", skip_header=2).T
    bv_obs = np.concatenate((bv_obs, data[0]))
    bv_err = np.concatenate((bv_err, data[1]))
    log_period_obs = np.concatenate((log_period_obs, np.log10(data[2])))
    log_period_err = np.concatenate((log_period_err, (log_errorbar(data[2], data[3], data[3]))[0]))
    log_age_obs = np.concatenate((log_age_obs, np.log10(data[4])))
    log_age_err = np.concatenate((log_age_err, (log_errorbar(data[4], data[5], data[5]))[0]))

    # 3d plot
    pl.clf()
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(10**log_period_obs, bv_obs, 10**log_age_obs, c = 'b', marker = 'o')
    ax.set_xlabel('Rotational period (days)')
    ax.set_ylabel('bv')
    ax.set_zlabel('Age (Gyr)')
    pl.show()

    raw_input('enter')

    # plot period vs age
    pl.clf()
    pl.errorbar(10**log_age_obs, 10**log_period_obs, xerr=10**log_age_err, yerr=10**log_period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')

    log_age_plot = np.linspace(0, max(log_age_obs))
    pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 6000.), 'r-')
    pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 5500.), 'm-')
    pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 5000.), 'b-')
    pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 4000.), 'c-')
    pl.xlabel('$\mathrm{Age~(Gyr)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    pl.savefig("init_bv_ap")

    # plot period vs teff
    pl.clf()
    pl.errorbar(bv_obs, 10**log_period_obs, xerr=bv_err, yerr=10**log_period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')
    # pl.scatter(bv_obs, 10**log_period_obs, c = logg_obs, s=50, vmin=min(logg_obs[logg_obs>0]), vmax=max(logg_obs), zorder=2)
    bv_plot = np.linspace(min(bv_obs), max(bv_obs))
    pl.plot(bv_plot, 10**log_period_model(par_true, np.log10(1.), bv_plot), 'r-')
    pl.plot(bv_plot, 10**log_period_model(par_true, np.log10(2.), bv_plot), 'm-')
    pl.plot(bv_plot, 10**log_period_model(par_true, np.log10(5.), bv_plot), 'b-')
    pl.plot(bv_plot, 10**log_period_model(par_true, np.log10(10.), bv_plot), 'c-')
    pl.xlabel('$\mathrm{T_{eff}~(K)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    # pl.colorbar()
    pl.savefig("init_bv_p")

    raw_input('enter')

    # Now generate samples
    nsamp = 100
    log_age_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_age_obs, log_age_err)])
    bv_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(bv_obs, bv_err)])
    logg_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(logg_obs, logg_err)])
    log_period_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_period_obs, log_period_err)])
    # FIXME: asymmetric errorbars for age and logg

    # calculate ms turnoff coeffs
    coeffs = MS_poly()

    print 'initial likelihood = ', lnlike(par_true, log_age_samp, bv_samp, \
            log_period_samp, logg_samp, bv_obs, bv_err, log_period_obs, log_period_err, \
            logg_obs, logg_err, coeffs, c)

    nwalkers, ndim = 32, len(par_true)
    p0 = [par_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (log_age_samp, bv_samp, log_period_samp, logg_samp, bv_obs, \
            bv_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, c)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, 500)
    sampler.reset()
    print("Production run")
    sampler.run_mcmc(p0, 10000)

    print("Plotting traces")
    pl.figure()
    for i in range(ndim):
        pl.clf()
        pl.axhline(par_true[i], color = "r")
        pl.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3)
        pl.savefig("{0}.png".format(i))

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
    fig.savefig("triangle_bv%s.png" %fname)

    mcmc_result = [mcmc_result[0], mcmc_result[1], mcmc_result[2], c, \
            mcmc_result[3], mcmc_result[4]]

    # plot result
    pl.clf()
    pl.errorbar(10**log_age_obs, 10**log_period_obs, xerr=log_age_err, yerr=10**log_period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')
    log_age_plot = np.linspace(0, max(log_age_obs))
    pl.plot(10**log_age_plot, 10**log_period_model(mcmc_result, log_age_plot, .45), 'r-')
    pl.plot(10**log_age_plot, 10**log_period_model(mcmc_result, log_age_plot, 1.), 'b-')
    pl.plot(10**log_age_plot, 10**log_period_model(mcmc_result, log_age_plot, .8), 'm-')
    pl.plot(10**log_age_plot, 10**log_period_model(mcmc_result, log_age_plot, .6), 'c-')
    pl.xlabel('$\mathrm{Age~(Gyr)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    pl.savefig("result_bv%s" %fname)

    # plot period vs teff
    pl.clf()
    pl.errorbar(bv_obs, 10**log_period_obs, xerr=bv_err, yerr=10**log_period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')
    bv_plot = np.linspace(min(bv_obs), max(bv_obs))
    pl.plot(bv_plot, 10**log_period_model(mcmc_result, np.log10(1.), bv_plot), color=ocols[0],\
             linestyle='-')
    pl.plot(bv_plot, 10**log_period_model(mcmc_result, np.log10(2.), bv_plot), color=ocols[1],\
             linestyle='-')
    pl.plot(bv_plot, 10**log_period_model(mcmc_result, np.log10(5.), bv_plot), color=ocols[2],\
             linestyle='-')
    pl.plot(bv_plot, 10**log_period_model(mcmc_result, np.log10(10.), bv_plot), color=ocols[3],\
             linestyle='-')
    pl.xlabel('$\mathrm{T_{eff}~(K)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    pl.savefig("result_bv%s" %fname)

if __name__ == "__main__":

    MCMC('run')
