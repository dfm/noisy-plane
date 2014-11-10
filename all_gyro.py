import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as spo
import emcee
import triangle
from all_plotting import load_dat, stratifiedkfold
import h5py
from subgiants import MS_poly
from mpl_toolkits.mplot3d import Axes3D
import datetime
from gyro_like import lnlike, period_model
from no_sampling_like import nslnlike

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
pl.rcParams.update(plotpar)

def lnprior(m):
#     if 0. < m[0] < .6 and 0. < m[1] < 1. and 0. < m[2] < 1. \
#     if 0. < m[0] < .65 and .5 < m[1] < 1. and 0. < m[2] < 1. \
    if -10. < m[0] < 10. and .0< m[1] < 1. and 0. < m[2] < 1. \
            and 0 < m[3] < 30. and 0 < m[4] < 100.\
            and 0 < m[5] < 30. and 0 < m[6] < 100.\
            and 0 < m[7] < 30. and 0 < m[8] < 100.\
            and 0. < m[9] < 1.:
        return 0.0
    return -np.inf

def lnprob(m, age_samp, bv_samp, period_samp, logg_samp, age_obs, age_err, \
        bv_obs, bv_err, period_obs, period_err, logg_obs, \
        logg_err, c, sampling):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    if sampling:
        return lp + lnlike(m, age_samp, bv_samp, \
                period_samp, logg_samp, age_obs, age_err, bv_obs, bv_err, period_obs, \
                period_err, logg_obs, logg_err, c)
    else:
        return lp + nslnlike(m, age_samp, bv_samp, \
                period_samp, logg_samp, age_obs, age_err, bv_obs, bv_err, period_obs, \
                period_err, logg_obs, logg_err, c)

def MCMC(fname, n, c, train, cv, sampling):

    # load MAP values
    try:
        params = np.genfromtxt('parameters%s.txt'%fname).T
        par_true = params[0]
        print 'initialising with MAP values'
        print par_true
    except:
#         par_true = [0.7725, 0.5189, .601, 5., 10., \ # Barnes
        par_true = [0.6, 0.5189, .601, 5., 10., \
                8., 30., 9., 5., .67] # better initialisation
        print par_true

    # load real data
    age_obs, age_err, age_errp, age_errm, period_obs, period_err, bv_obs, bv_err, \
            logg_obs, logg_err, logg_errp, logg_errm, flag = load_dat(fname, train, cv)

    pl.clf()
    pl.errorbar(age_obs, period_obs, xerr=age_err, yerr=period_err, fmt='k.',
                 capsize=0, ecolor='.8')
#     pl.show()

    # Now generate samples
    nsamp = 100 # FIXME
    np.random.seed(12)
    age_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(age_obs, age_err)])
    np.random.seed(12)
    bv_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(bv_obs, bv_err)])
    np.random.seed(12)
    logg_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(logg_obs, logg_err)])
    np.random.seed(12)
    period_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(period_obs, period_err)])

    if sampling:
        print 'initial likelihood = ', lnlike(par_true, age_samp, bv_samp, \
                period_samp, logg_samp, age_obs, age_err, bv_obs, bv_err, period_obs, period_err, \
                logg_obs, logg_err, c)
    else:
        print 'initial likelihood = ', nslnlike(par_true, age_samp, bv_samp, \
                period_samp, logg_samp, age_obs, age_err, bv_obs, bv_err, period_obs, period_err, \
                logg_obs, logg_err, c)

    nwalkers, ndim = 32, len(par_true)
    p0 = [par_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (age_samp, bv_samp, period_samp, logg_samp, age_obs, age_err, bv_obs, \
            bv_err, period_obs, period_err, logg_obs, logg_err, c, sampling)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)

    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, 5000)
#     p0, lp, state = sampler.run_mcmc(p0, 300)
    sampler.reset()
    print("Production run")
#     nstep = 3000
#     nruns = 500.
    nstep = 20000
    nruns = 2000.

    for j in range(int(nstep/nruns)):

        print fname, n
        print datetime.datetime.now()
        print 'run', j
        p0, lp, state = sampler.run_mcmc(p0, nruns)

        flat = sampler.chain[:, 50:, :].reshape((-1, ndim))
        mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                          zip(*np.percentile(flat, [16, 50, 84], axis=0)))
        mres = np.array(mcmc_result)[:, 0]
        print 'mcmc_result = ', mres

        print "saving samples"
        f = h5py.File("samples_%s" %fname, "w")
        data = f.create_dataset("samples", np.shape(sampler.chain))
        data[:,:] = np.array(sampler.chain)
        f.close()

        samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

        likelihood = lnlike(mres, age_samp, bv_samp, period_samp,
                            logg_samp, age_obs, age_err, bv_obs, bv_err,
                            period_obs, period_err, logg_obs, logg_err, c)
        print 'likelihood = ', likelihood
#         np.savetxt('likelihood%s%s.txt'%(n, fname), likelihood)

    # save parameters and print to screen
    print 'initial values', par_true
    np.savetxt("parameters%s%s.txt" %(n, fname), np.array(mcmc_result))
    mcmc_result = np.array(mcmc_result)[:, 0]
    mcmc_result = [mcmc_result[0], mcmc_result[1], mcmc_result[2], c, \
            mcmc_result[3], mcmc_result[4]]

    # save samples
    f = h5py.File("samples_%s%s" %(n, fname), "w")
    data = f.create_dataset("samples", np.shape(sampler.chain))
    data[:,:] = np.array(sampler.chain)
    f.close()

if __name__ == "__main__":

    # proper runs
    fname = 'ACHF45_old'

    print fname, "fname"
    MCMC(fname, '_', .45, False, False, True)

#     cross_v = False
#     if cross_v:
#         # cross validation
#         folds = 5; n=0
#         skf = stratifiedkfold(folds, fname)
#         scores = []
#         for train, test in skf:
#             if n>0:
#                 MCMC(fname, n, .45, train, cross_v, True)
#             scores.append(scoring(fname, n, test))
#             print scores[n]
#             np.savetxt('train%s_%s.txt'%(n, fname), train)
#             np.savetxt('test%s_%s.txt'%(n, fname), test)
#             n+=1
#         np.savetxt('scores_%s.txt'%fname, scores)
#     else:
#         MCMC(fname, '_', .45, False, False, True)
