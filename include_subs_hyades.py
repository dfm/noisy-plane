import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as spo
import emcee
import triangle
from plotting import load_dat, log_errorbar
import pretty5
from load_hyades import hya_load

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
pl.rcParams.update(plotpar)

def log_period_model(par, log_age, bv):
    c = .4
    return par[0] + par[1] * log_age + par[2] * np.log10(bv - c)

def lnprior(m):
    if -10. < m[0] < 10. and 0. < m[1] < 10. and 0. < m[2] < 10.:
        return 0.0
    return -np.inf

def lnlike(par, log_age_samp, c_samp, log_period_samp, \
               c_obs, c_err, log_period_obs, log_period_err):
    nobs,nsamp = log_age_samp.shape
    log_period_pred = log_period_model(par[:4], log_age_samp, c_samp)
    ll = np.zeros(nobs)
    c = .4
    ll = np.zeros(nobs)
    for i in np.arange(nobs):

        # cool MS stars
        l1 = (c_samp[i,:] > c)
        if l1.sum() > 0:
            like1 = \
                np.exp(-((log_period_obs[i] - log_period_pred[i,l1])/2.0/log_period_err[i])**2) \
                / log_period_err[i]
            lik1 = np.sum(like1) / float(l1.sum())
        else:
            lik1 = 0.0

        ll[i] = np.log10(lik1)
    return np.sum(ll)

def lnprob(m, log_age_samp, c_samp, log_period_samp, \
        c_obs, c_err, log_period_obs, log_period_err):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m, log_age_samp, c_samp, \
            log_period_samp, c_obs, c_err, log_period_obs, log_period_err)

# log(a), n, beta, Y, V, Z, U
true_pars = [np.log10(0.7725), 0.5189, 0.601]
plot_pars = [np.log10(0.7725), 0.5189, 0.601, .4]

par_true = true_pars

# load data
period_obs, c_obs, age_obs, period_err, c_err, age_err = hya_load()
log_period_obs = np.log10(period_obs)
log_age_obs = np.log10(age_obs)
log_period_err = log_errorbar(period_obs, period_err, period_err)[0]
log_age_err = log_errorbar(age_obs, age_err, age_err)[0]

# convert to Myr
log_age_obs += 3.
log_age_err += 3.

# plot period vs age
pl.clf()
pl.errorbar(10**log_age_obs, 10**log_period_obs, xerr=10**log_age_err, \
        yerr=10**log_period_err, fmt='k.', capsize = 0, ecolor = '.7')

par_true = plot_pars
log_age_plot = np.linspace(0, max(log_age_obs))
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, .5), 'r-')
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, .7), 'b-')
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, .9), 'm-')
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 1.1), 'c-')
pl.xlabel('$\mathrm{Age~(Gyr)}$')
pl.ylabel('$P_{rot}~\mathrm{(days)}$')
pl.savefig("init")

# plot period vs teff
pl.clf()
pl.errorbar(c_obs, 10**log_period_obs, xerr=c_err, yerr=10**log_period_err, \
        fmt='k.', capsize = 0, ecolor = '.7')
c_plot = np.linspace(min(c_obs), max(c_obs))
pl.plot(c_plot, 10**log_period_model(par_true, np.log10(1000.), c_plot), 'r-')
pl.plot(c_plot, 10**log_period_model(par_true, np.log10(2000.), c_plot), 'b-')
pl.plot(c_plot, 10**log_period_model(par_true, np.log10(5000.), c_plot), 'm-')
pl.plot(c_plot, 10**log_period_model(par_true, np.log10(10000.), c_plot), 'c-')
pl.xlabel('$\mathrm{T_{eff}~(K)}$')
pl.ylabel('$P_{rot}~\mathrm{(days)}$')
pl.savefig("init_teff")

plots = pretty5.plotting()
plots.p_vs_t(par_true)
raw_input('enter')

par_true = true_pars

# Now generate samples
nsamp = 100
log_age_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_age_obs, log_age_err)])
c_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(c_obs, c_err)])
log_period_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_period_obs, log_period_err)])
# FIXME: asymmetric errorbars for age

print 'intial likelihood = ', lnlike(par_true, log_age_samp, c_samp, \
        log_period_samp, c_obs, c_err, log_period_obs, log_period_err)

nwalkers, ndim = 32, len(par_true)
p0 = [par_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
args = (log_age_samp, c_samp, log_period_samp, c_obs, c_err, log_period_obs, log_period_err)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

print("Burn-in")
p0, lp, state = sampler.run_mcmc(p0, 500)
sampler.reset()
print("Production run")
sampler.run_mcmc(p0, 2000)

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
fig_labels = ["$log(a)$", "$n$", "$b$"]
fig = triangle.corner(sampler.flatchain, truths=mcmc_result, labels=fig_labels[:len(par_true)])
fig.savefig("triangle.png")

mcmc_result = [mcmc_result[0], mcmc_result[1], mcmc_result[2], .4]

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
pl.savefig("result")

# plot period vs teff
pl.clf()
pl.errorbar(c_obs, 10**log_period_obs, xerr=c_err, yerr=log_period_err, fmt='k.', \
        capsize = 0, ecolor = '.7')
c_plot = np.linspace(min(c_obs), max(c_obs))
pl.plot(c_plot, 10**log_period_model(mcmc_result, np.log10(1.), c_plot), color=ocols[0],\
         linestyle='-')
pl.plot(c_plot, 10**log_period_model(mcmc_result, np.log10(2.), c_plot), color=ocols[1],\
         linestyle='-')
pl.plot(c_plot, 10**log_period_model(mcmc_result, np.log10(5.), c_plot), color=ocols[2],\
         linestyle='-')
pl.plot(c_plot, 10**log_period_model(mcmc_result, np.log10(10.), c_plot), color=ocols[3],\
         linestyle='-')
pl.xlabel('$\mathrm{T_{eff}~(K)}$')
pl.ylabel('$P_{rot}~\mathrm{(days)}$')
pl.savefig("result_teff")

plots = pretty5.plotting()
plots.p_vs_t(mcmc_result)
