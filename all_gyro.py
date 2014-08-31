import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as spo
import emcee
import triangle
from all_plotting import load_dat
from gyro_like import lnlike, period_model
import h5py
from subgiants import MS_poly
from mpl_toolkits.mplot3d import Axes3D
import datetime

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
pl.rcParams.update(plotpar)

def lnprior(m):
    if -10. < m[0] < 10. and 0. < m[1] < 1. and 0. < m[2] < 1. \
            and 0 < m[3] < 30. and 0 < m[4] < 100.\
            and 0 < m[5] < 30. and 0 < m[6] < 100.\
            and 0 < m[7] < 30. and 0 < m[8] < 100.\
            and 0. < m[9] < 1.:
        return 0.0
    return -np.inf

def lnprob(m, age_samp, bv_samp, period_samp, logg_samp, age_obs, age_err, \
        bv_obs, bv_err, period_obs, period_err, logg_obs, \
        logg_err, c):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m, age_samp, bv_samp, \
            period_samp, logg_samp, age_obs, age_err, bv_obs, bv_err, period_obs, \
            period_err, logg_obs, logg_err, c)

def MCMC(fname, c):

    # load MAP values
    try:
        params = np.genfromtxt('parameters%s.txt'%fname).T
        par_true = params[0]
        print par_true
    except:
        par_true = [0.7725, 0.5189, .601, 5., 10., \
                8., 30., 9., 5., .67] # better initialisation
        print par_true

    # load real data
    age_obs, age_err, age_errp, age_errm, period_obs, period_err, bv_obs, bv_err, \
            logg_obs, logg_err, logg_errp, logg_errm = load_dat(fname)

    # Now generate samples
    nsamp = 100
    age_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(age_obs, age_err)])
    bv_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(bv_obs, bv_err)])
    logg_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(logg_obs, logg_err)])
    period_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(period_obs, period_err)])

    print 'initial likelihood = ', lnlike(par_true, age_samp, bv_samp, \
            period_samp, logg_samp, age_obs, age_err, bv_obs, bv_err, period_obs, period_err, \
            logg_obs, logg_err, c)

    nwalkers, ndim = 32, len(par_true)
    p0 = [par_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    args = (age_samp, bv_samp, period_samp, logg_samp, age_obs, age_err, bv_obs, \
            bv_err, period_obs, period_err, logg_obs, logg_err, c)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)

    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, 2000)
#     p0, lp, state = sampler.run_mcmc(p0, 300)
    sampler.reset()
    print("Production run")
    nstep = 50000
    nruns = 2000.
#     nruns = 500.

    for j in range(int(nstep/nruns)):

        print fname
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
        # Flatten chain
        samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    # save parameters and print to screen
    print 'initial values', par_true
    np.savetxt("parameters%s.txt" %fname, np.array(mcmc_result))
    mcmc_result = np.array(mcmc_result)[:, 0]
    mcmc_result = [mcmc_result[0], mcmc_result[1], mcmc_result[2], c, \
            mcmc_result[3], mcmc_result[4]]

    # save samples
    print sampler.chain
    f = h5py.File("samples_%s" %fname, "w")
    data = f.create_dataset("samples", np.shape(sampler.chain))
    data[:,:] = np.array(sampler.chain)
    f.close()

if __name__ == "__main__":

    MCMC('PF5', .5)
