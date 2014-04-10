import numpy as np
import pylab as pl
import scipy.optimize as spo
import emcee
import triangle
from plotting import load_dat

TEMP_MAX = 7500
TEMP_MIN = 3000

def log_period_model(par, log_age, temp):
    return par[0] + par[1] * log_age + par[2] * np.log10(par[3] - temp)

def lnprior(m):
    if -10. < m[0] < 10. and -10. < m[1] < 10. and -10. < m[2] < 10. and 3000. < m[3] < 10000.:
        return 0.0
    return -np.inf

def negloglike(par, log_age_samp, temp_samp, log_period_samp, \
               temp_obs, temp_err, log_period_obs, log_period_err):
    nobs,nsamp = log_age_samp.shape
    log_period_pred = log_period_model(par[:4], log_age_samp, temp_samp)
    ll = np.zeros(nobs)
    temp_Kraft = par[3]
    A = max(0,(TEMP_MAX- temp_Kraft) / float(TEMP_MAX - TEMP_MIN))
    ll = np.zeros(nobs)
    for i in np.arange(nobs):
        l1 = temp_samp[i,:] < temp_Kraft
        if l1.sum() > 0:
            like1 = \
                np.exp(-((log_period_obs[i] - log_period_pred[i,l1])/2.0/log_period_err[i])**2) \
                / log_period_err[i]
            lik1 = np.sum(like1) / float(l1.sum())
        else:
            lik1 = 0.0
        l2 = l1 == False
        if l2.sum() > 0:
            like2 = A * \
                np.exp(-((temp_obs[i] - temp_samp[i,l2])/2.0/temp_err[i])**2) \
                / temp_err[i]
            lik2 = np.sum(like2) / float(l2.sum())
        else:
            lik2 = 0.0
        ll[i] = np.log10(lik1 + lik2)
    return -np.sum(ll)

def lnlike(par, log_age_samp, temp_samp, log_period_samp, \
               temp_obs, temp_err, log_period_obs, log_period_err):
    nobs,nsamp = log_age_samp.shape
    log_period_pred = log_period_model(par[:4], log_age_samp, temp_samp)
    ll = np.zeros(nobs)
    temp_Kraft = par[3]
    A = max(0,(TEMP_MAX- temp_Kraft) / float(TEMP_MAX - TEMP_MIN))
    ll = np.zeros(nobs)
    for i in np.arange(nobs):
        l1 = temp_samp[i,:] < temp_Kraft
        if l1.sum() > 0:
            like1 = \
                np.exp(-((log_period_obs[i] - log_period_pred[i,l1])/2.0/log_period_err[i])**2) \
                / log_period_err[i]
            lik1 = np.sum(like1) / float(l1.sum())
        else:
            lik1 = 0.0
        l2 = l1 == False
        if l2.sum() > 0:
            like2 = A * \
                np.exp(-((temp_obs[i] - temp_samp[i,l2])/2.0/temp_err[i])**2) \
                / temp_err[i]
            lik2 = np.sum(like2) / float(l2.sum())
        else:
            lik2 = 0.0
        ll[i] = np.log10(lik1 + lik2)
    return np.sum(ll)

def lnprob(m, log_age_samp, temp_samp, log_period_samp, \
        temp_obs, temp_err, log_period_obs, log_period_err):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m, log_age_samp, temp_samp, \
            log_period_samp, temp_obs, temp_err, log_period_obs, log_period_err)

# load real data
log_period_obs, temp_obs, log_age_obs, period_err, temp_err, age_err = load_dat()
# need to add errorbars later

# Generate set of fake observations
nobs = len(log_period_obs)
log_age_true = np.random.uniform(0,1,nobs)
log_age_true = np.zeros(nobs)
log_age_true[:20] = 0.0
log_age_true[20:40] = 0.3
log_age_true[40:] = 1.0

temp_true = np.random.uniform(3500,7000,nobs)

# First create noise-free values
par_true = [np.log10(0.7725), 0.5189, -0.06, 6300.]
log_period_true = log_period_model(par_true,log_age_true,temp_true)
l = np.isfinite(log_period_true) == False
n = l.sum()
log_period_true[l] = np.random.uniform(0,1,n)

# Then add noise
log_age_err = np.zeros(nobs) + 0.05
# log_age_obs = np.random.normal(log_age_true, log_age_err)
temp_err = np.zeros(nobs) + 100
temp_obs = np.random.normal(temp_true, temp_err)
log_period_err = np.zeros(nobs) + 0.05
# log_period_obs = np.random.normal(log_period_true, log_period_err)


# Now generate samples
nsamp = 100
log_age_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_age_obs, log_age_err)])
temp_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(temp_obs, temp_err)])
log_period_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_period_obs, log_period_err)])

print par_true, 'par'
par_fit = spo.fmin(negloglike, par_true, \
                   args = (log_age_samp, temp_samp, log_period_samp, \
                           temp_obs, temp_err, log_period_obs, \
                           log_period_err))
print par_true
print par_fit

pl.clf()
pl.plot(temp_samp, log_period_samp, ',', c = 'grey', mec = 'grey', alpha = 0.5)
pl.errorbar(temp_obs, log_period_obs, xerr = temp_err, yerr = log_period_err,\
              fmt = 'b.', mec = 'b', capsize = 0)
pl.plot(temp_true, log_period_true, 'r.', mec = 'r')
pl.xlim(7500,3000)
pl.xlabel('$T_{\mathrm{eff}}$ (K)')
pl.ylabel('$\log_{10} P_{\mathrm{rot}}$ (days)')
xx = np.r_[TEMP_MIN:TEMP_MAX:10]
yy = np.array([0, 0.3, 1.0])
for i in np.arange(len(yy)):
    zz = log_period_model(par_fit,yy[i],xx)
    pl.plot(xx, zz, 'g--',lw=2)
pl.savefig('SuzGyroTest.png')

print 'intial likelihood = ', lnlike(par_true, log_age_samp, temp_samp, \
        log_period_samp, temp_obs, temp_err, log_period_obs, log_period_err)

nwalkers, ndim = 32, len(par_true)
p0 = [par_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
args = (log_age_samp, temp_samp, log_period_samp, temp_obs, temp_err, log_period_obs, log_period_err)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

print("Burn-in")
p0, lp, state = sampler.run_mcmc(p0, 500)
sampler.reset()
print("Production run")
sampler.run_mcmc(p0, 2000)

print("Making triangle plot")
fig_labels = ["$n$", "$a$", "$b$", "$T_K$"]
fig = triangle.corner(sampler.flatchain, truths=par_true, labels=fig_labels[:len(par_true)])
fig.savefig("triangle.png")

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
