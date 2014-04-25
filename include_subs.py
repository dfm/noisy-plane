import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as spo
import emcee
import triangle
from plotting import load_dat
import pretty5

ocols = ['#FF9933','#66CCCC' , '#FF33CC', '#3399FF', '#CC0066', '#99CC99', '#9933FF', '#CC0000']
plotpar = {'axes.labelsize': 20,
           'text.fontsize': 20,
           'legend.fontsize': 15,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
pl.rcParams.update(plotpar)

def log_period_model(par, log_age, temp):
    Tk = 6250
    return par[0] + par[1] * log_age + par[2] * np.log10(Tk - temp)

def lnprior(m):
    if -10. < m[0] < 10. and 0. < m[1] < 10. and 0. < m[2] < 10. \
            and 0 < m[3] < np.log10(30.) and 0 < m[4] < np.log10(100.)\
            and 0 < m[5] < np.log10(30.) and 0 < m[6] < np.log10(100.):
        return 0.0
    return -np.inf

def lnlike(par, log_age_samp, temp_samp, log_period_samp, \
               temp_obs, temp_err, log_period_obs, log_period_err):
    nobs,nsamp = log_age_samp.shape
    log_period_pred = log_period_model(par[:4], log_age_samp, temp_samp)
    ll = np.zeros(nobs)
    temp_Kraft = 6250
    logg_cut = 4.
    Y, V = par[3], par[4]
    Z, U = par[5], par[6]
    ll = np.zeros(nobs)
    for i in np.arange(nobs):

        # cool MS stars
        l1 = (temp_samp[i,:] < temp_Kraft) * (logg_samp[i,:] > 4.)
        if l1.sum() > 0:
            like1 = \
                np.exp(-((log_period_obs[i] - log_period_pred[i,l1])/2.0/log_period_err[i])**2) \
                / log_period_err[i]
            lik1 = np.sum(like1) / float(l1.sum())
        else:
            lik1 = 0.0

        # hot MS stars
        l2 = (temp_samp[i,:] > temp_Kraft) * (logg_samp[i,:] > 4.)
        if l2.sum() > 0:
            like2 = np.exp(-((log_period_obs[i] - Y)**2/2.0**2/((log_period_err[i])**2+V))) \
                / (log_period_err[i]+V)
            lik2 = np.sum(like2) / float(l2.sum())
        else:
            lik2 = 0.0

        # subgiants
        l3 = logg_samp[i,:] < 4.
        if l3.sum() > 0:
            like3 = np.exp(-((log_period_obs[i] - Z)**2/2.0**2/((log_period_err[i])**2+U))) \
                / (log_period_err[i]+U)
            lik3 = np.sum(like3) / float(l3.sum())
        else:
            lik3 = 0.0

        ll[i] = np.log10(lik1 + lik2 + lik3)
    return np.sum(ll)

def lnprob(m, log_age_samp, temp_samp, log_period_samp, \
        temp_obs, temp_err, log_period_obs, log_period_err):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m, log_age_samp, temp_samp, \
            log_period_samp, temp_obs, temp_err, log_period_obs, log_period_err)

# log(a), n, beta, Y, V, Z, U
true_pars = [np.log10(0.7725), 0.5189, .2, np.log10(5.), np.log10(10.), \
        np.log10(10.), np.log10(10.)]
plot_pars = [np.log10(0.7725), 0.5189, .2, 6250, np.log10(5.), np.log10(10.)]

par_true = true_pars

# load real data
log_period_obs, temp_obs, log_age_obs, log_period_err, temp_err, log_age_err, \
        logg_obs, logg_err = load_dat()

# # Generate set of fake observations
# nobs = len(log_period_obs)
# nobs = 60
# log_age_true = np.random.uniform(0,1,nobs)
# log_age_true = np.zeros(nobs)
# log_age_true[:20] = 0.0
# log_age_true[20:40] = 0.3
# log_age_true[40:] = 1.0
#
# temp_true = np.random.uniform(3500,7000,nobs)
# logg_true = np.random.uniform(3.5,4.5,nobs)
#
# # First create noise-free values
# log_period_true = log_period_model(par_true,log_age_true,temp_true)
# l = np.isfinite(log_period_true) == False
# n = l.sum()
# log_period_true[l] = np.random.uniform(0,1,n)
#
# # Then add noise
# log_age_err = np.zeros(nobs) + 0.05
# log_period_err = np.zeros(nobs) + 0.05
# temp_err = np.zeros(nobs) + 100
# logg_err = np.zeros(nobs) + 0.05
# log_age_err = np.ones(nobs)*np.mean(log_age_err)
# log_age_obs = np.random.normal(log_age_true, log_age_err)
# temp_err = np.ones(nobs)*np.mean(temp_err)
# temp_obs = np.random.normal(temp_true, temp_err)
# log_period_err = np.ones(nobs)*np.mean(log_period_err)
# log_period_obs = np.random.normal(log_period_true, log_period_err)
# logg_obs = np.random.normal(logg_true, logg_err)

# plot period vs age
pl.clf()
pl.errorbar(10**log_age_obs, 10**log_period_obs, xerr=log_age_err, yerr=10**log_period_err, fmt='k.', \
        capsize = 0, ecolor = '.7')

par_true = plot_pars
log_age_plot = np.linspace(0, max(log_age_obs))
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 6000.), 'r-')
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 4000.), 'b-')
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 5000.), 'm-')
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 4500.), 'c-')
pl.xlabel('$\mathrm{Age~(Gyr)}$')
pl.ylabel('$P_{rot}~\mathrm{(days)}$')
pl.savefig("init")

# plot period vs teff
pl.clf()
pl.errorbar(temp_obs, 10**log_period_obs, xerr=temp_err, yerr=10**log_period_err, fmt='k.', \
        capsize = 0, ecolor = '.7')
temp_plot = np.linspace(min(temp_obs), max(temp_obs))
pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(1.), temp_plot), 'r-')
pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(2.), temp_plot), 'b-')
pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(5.), temp_plot), 'm-')
pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(10.), temp_plot), 'c-')
pl.xlim(pl.gca().get_xlim()[::-1])
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
temp_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(temp_obs, temp_err)])
logg_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(logg_obs, logg_err)])
log_period_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_period_obs, log_period_err)])
# FIXME: asymmetric errorbars for age and logg

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
sampler.run_mcmc(p0, 5000)

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
fig_labels = ["$log(a)$", "$n$", "$b$", "$Y$", "$V$", "$Z$", "$U$"]
fig = triangle.corner(sampler.flatchain, truths=mcmc_result, labels=fig_labels[:len(par_true)])
fig.savefig("triangle.png")

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
pl.savefig("result")

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
pl.savefig("result_teff")

plots = pretty5.plotting()
plots.p_vs_t(mcmc_result)
