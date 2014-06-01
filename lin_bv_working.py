# Try adjusting Kraft break?

import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as spo
import emcee
import triangle
from plotting import load_dat, log_errorbar
import pretty5
from subgiants import MS_poly
from lin_bv_likes import lnlike
from teff_bv import teff2bv
import plotting_working as pw

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
pl.rcParams.update(plotpar)

c = .4

def period_model(par, age, bv):
    log_age = np.log10(age)
    log_period = par[0] + par[1]*log_age + np.log10(bv-c) * par[2]
    return 10**log_period

def lnprior(m):
    if -10. < m[0] < 10. and .3 < m[1] < .8 and 0. < m[2] < 1. \
            and 0 < m[3] < 30. and 0 < m[4] < 100.\
            and 0 < m[5] < 30. and 0 < m[6] < 100.\
            and 0 < m[7] < 30. and 0 < m[8] < 100.\
            and 0. < m[9] < 1.:
        return 0.0
#         return -0.01*(m[1]+.5189)**2 #-0.5*(m[3]+.2)**2
    return -np.inf

def lnprob(m, age_samp, bv_samp, period_samp, logg_samp, \
        bv_obs, bv_err, period_obs, period_err, logg_obs, \
        logg_err, coeffs, c):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m, age_samp, bv_samp, \
            period_samp, logg_samp, bv_obs, bv_err, period_obs, \
            period_err, logg_obs, logg_err, coeffs, c)

def MCMC(fname):
    # a, n, beta, Y, V, Z, U
#     par_true = [0.7725, 0.5189, .601, 5., 10., \
#             10., 10., 1.5, 5., .5]
    par_true = [0.7725, 0.5189, .601, 5., 10., \
            8., 5., 9., 3.5, .67] # better initialisation

    # load real data
    log_period_obs, bv_obs, log_age_obs, log_period_err, bv_err, log_age_err, log_age_errp, log_age_errm, \
            logg_obs, logg_err, logg_errp, logg_errm, age_obs, age_errp, age_errm, age_err, period_obs, period_err = load_dat()

    # load real data
    log_period_obs2, temp_obs, log_age_obs2, log_period_err2, temp_err2, log_age_err2, \
            log_age_errp2, log_age_errm2, \
            logg_obs2, logg_err2, logg_errp2, logg_errm2, age_obs2, age_errp2, age_errm2, \
            period_obs2, period_err2 = pw.load_dat()

    # 3d colour plot
    pl.clf()
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(period_obs, bv_obs, age_obs, c = 'b', marker = 'o')
    ax.set_xlabel('Rotational period (days)')
    ax.set_ylabel('bv')
    ax.set_zlabel('Age (Gyr)')
#     fx, fy, fz = period_obs, bv_obs, age_obs
#     xerror, yerror, zerror = period_err, bv_err, age_err
#     #plot errorbars
#     for i in np.arange(0, len(fx)):
#         ax.plot([fx[i]+xerror[i], fx[i]-xerror[i]], [fy[i], fy[i]], [fz[i], fz[i]], marker="_")
#         ax.plot([fx[i], fx[i]], [fy[i]+yerror[i], fy[i]-yerror[i]], [fz[i], fz[i]], marker="_")
#         ax.plot([fx[i], fx[i]], [fy[i], fy[i]], [fz[i]+zerror[i], fz[i]-zerror[i]], marker="_")
    a_surf = np.linspace(.65, 15, 100)
    c_surf = np.linspace(.4, 1.4, 100)
    a_surf, c_surf = np.meshgrid(a_surf, c_surf)
    p_surf = period_model(par_true, a_surf, c_surf)
    ax.plot_surface(p_surf, c_surf, a_surf, alpha = 0.0)
#     pl.show()

    # plot period vs age
    pl.clf()
    pl.errorbar(age_obs, period_obs, xerr=age_err, yerr=period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')

    age_plot = np.linspace(0, max(age_obs))
    pl.plot(age_plot, period_model(par_true, age_plot, .45), 'r-')
    pl.plot(age_plot, period_model(par_true, age_plot, .6), 'm-')
    pl.plot(age_plot, period_model(par_true, age_plot, .8), 'b-')
    pl.plot(age_plot, period_model(par_true, age_plot, 1.), 'c-')
    pl.xlabel('$\mathrm{Age~(Gyr)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    pl.savefig("init_bv_ap%s" %fname)

    # plot period vs teff
    pl.clf()
    pl.errorbar(bv_obs, period_obs, xerr=bv_err, yerr=period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')
    bv_plot = np.linspace(min(bv_obs), max(bv_obs))
    pl.plot(bv_plot, period_model(par_true, 1., bv_plot), 'r-')
    pl.plot(bv_plot, period_model(par_true, 2., bv_plot), 'm-')
    pl.plot(bv_plot, period_model(par_true, 5., bv_plot), 'b-')
    pl.plot(bv_plot, period_model(par_true, 10., bv_plot), 'c-')
    pl.xlabel('$\mathrm{T_{eff}~(K)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    pl.savefig("init_bv_p%s" %fname)

    # Now generate samples
    nsamp = 100
    age_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(age_obs, age_err)])

    # resample negative ages
#     for j in range(len(age_samp)):
#         while len(age_samp[j][age_samp[j]<0])>0:
#             age_samp[j][age_samp[j]<0] = [x0+xe*np.random.randn(len(age_samp[j][age_samp[j]<0])) \
#                     for x0, xe in zip(age_obs[j], age_err[j])]

    age_samp[age_samp<0] = 0.1
    bv_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(bv_obs, bv_err)])
    logg_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(logg_obs, logg_err)])
    period_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(period_obs, period_err)])

    # calculate ms turnoff coeffs
    coeffs = MS_poly()

    print 'initial likelihood = ', lnlike(par_true, age_samp, bv_samp, \
            period_samp, logg_samp, bv_obs, bv_err, period_obs, period_err, \
            logg_obs, logg_err, coeffs, c)

    nwalkers, ndim = 32, len(par_true)
    p0 = [par_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (age_samp, bv_samp, period_samp, logg_samp, bv_obs, \
            bv_err, period_obs, period_err, logg_obs, logg_err, coeffs, c)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)

    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, 2000)
    sampler.reset()
    print("Production run")
    nstep = 10000
    nruns = 2000.

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
    mcmc_result = [mcmc_result[0], mcmc_result[1], mcmc_result[2], c, \
            mcmc_result[3], mcmc_result[4]]
    np.savetxt("parameters%s.txt" %fname, mcmc_result)

    # plot result
    pl.clf()
    pl.errorbar(age_obs, period_obs, xerr=age_err, yerr=period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')
    age_plot = np.linspace(0, max(age_obs))
    pl.plot(age_plot, period_model(mcmc_result, age_plot, .45), 'r-')
    pl.plot(age_plot, period_model(mcmc_result, age_plot, 1.), 'b-')
    pl.plot(age_plot, period_model(mcmc_result, age_plot, .8), 'm-')
    pl.plot(age_plot, period_model(mcmc_result, age_plot, .6), 'c-')
    pl.xlabel('$\mathrm{Age~(Gyr)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    pl.savefig("result_bv%s" %fname)

    # plot period vs teff
    pl.clf()
    pl.errorbar(bv_obs, period_obs, xerr=bv_err, yerr=period_err, fmt='k.', \
            capsize = 0, ecolor = '.7')
    bv_plot = np.linspace(min(bv_obs), max(bv_obs))
    pl.plot(bv_plot, period_model(mcmc_result, 1., bv_plot), color=ocols[0],\
             linestyle='-')
    pl.plot(bv_plot, period_model(mcmc_result, 2., bv_plot), color=ocols[1],\
             linestyle='-')
    pl.plot(bv_plot, period_model(mcmc_result, 5., bv_plot), color=ocols[2],\
             linestyle='-')
    pl.plot(bv_plot, period_model(mcmc_result, 10., bv_plot), color=ocols[3],\
             linestyle='-')
    pl.xlabel('$\mathrm{T_{eff}~(K)}$')
    pl.ylabel('$P_{rot}~\mathrm{(days)}$')
    pl.savefig("result_bv%s" %fname)

if __name__ == "__main__":

    MCMC('4')
