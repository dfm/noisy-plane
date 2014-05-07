import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as spo
import emcee
import triangle
from plotting import load_dat
import pretty5
from subgiants import MS_poly
from lnlikes import lnlike

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
            and 0 < m[7] < np.log10(30.) and 0 < m[8] < np.log10(100.):
        return -.5*((m[1]-.5189)/.2)**2 -.5*((m[2]-.2)/.2)**2
#         return -.5*((m[1]-.5189)/.4)**2
#         return 0.0
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

# log(a), n, beta, Y, V, Z, U
true_pars = [np.log10(0.7725), 0.5189, .2, np.log10(5.), np.log10(10.), \
        np.log10(10.), np.log10(10.), np.log10(1.5), np.log10(5.)]
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
pl.errorbar(10**log_age_obs, 10**log_period_obs, xerr=10**log_age_err, yerr=10**log_period_err, fmt='k.', \
        capsize = 0, ecolor = '.7')

par_true = plot_pars
log_age_plot = np.linspace(0, max(log_age_obs))
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 6000.), 'r-')
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 5500.), 'm-')
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 5000.), 'b-')
pl.plot(10**log_age_plot, 10**log_period_model(par_true, log_age_plot, 4000.), 'c-')
pl.xlabel('$\mathrm{Age~(Gyr)}$')
pl.ylabel('$P_{rot}~\mathrm{(days)}$')
pl.savefig("init")

coeffs = MS_poly()
turnoff = np.polyval(coeffs, temp_obs)-.1
a = (temp_obs < 6250)*(logg_obs > .4) * (10**log_period_obs)/(10**(log_age_obs)**.5189) > 2.1
pl.clf()
pl.errorbar(10**log_age_obs, 10**log_period_obs/(6250-temp_obs)**0.2, xerr=10**log_age_err, yerr=10**log_period_err, fmt='k.', \
        capsize = 0, ecolor = '.7')
pl.plot(10**log_age_obs[a], 10**log_period_obs[a]/(6250-temp_obs[a])**.2, 'r.')
par_true = plot_pars
log_age_plot = np.linspace(0, max(log_age_obs))
pl.plot(10**log_age_plot, 10**(.5189*log_age_plot), 'r-')
pl.xlabel('$\mathrm{Age~(Gyr)}$')
pl.ylabel('$P_{rot}/(T_k-T_{eff})^b$')
pl.savefig("init2")

# plot period vs teff
pl.clf()
pl.errorbar(temp_obs, 10**log_period_obs, xerr=temp_err, yerr=10**log_period_err, fmt='k.', \
        capsize = 0, ecolor = '.7')
# pl.scatter(temp_obs, 10**log_period_obs, c = logg_obs, s=50, vmin=min(logg_obs[logg_obs>0]), vmax=max(logg_obs), zorder=2)
temp_plot = np.linspace(min(temp_obs), max(temp_obs))
pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(1.), temp_plot), 'r-')
pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(2.), temp_plot), 'm-')
pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(5.), temp_plot), 'b-')
pl.plot(temp_plot, 10**log_period_model(par_true, np.log10(10.), temp_plot), 'c-')
pl.xlim(pl.gca().get_xlim()[::-1])
pl.xlabel('$\mathrm{T_{eff}~(K)}$')
pl.ylabel('$P_{rot}~\mathrm{(days)}$')
# pl.colorbar()
pl.savefig("init_teff")

a = (temp_obs < 6250)*(logg_obs > .4) * (10**log_period_obs)/(10**(log_age_obs)**.5189) > 2.1
pl.clf()
pl.errorbar(temp_obs, 10**log_period_obs/(10**(log_age_obs)**.5189), xerr=temp_err, yerr=10**log_period_err, fmt='k.', \
        capsize = 0, ecolor = '.7', zorder=1)
# pl.scatter(temp_obs, 10**log_period_obs/(10**(log_age_obs)**.5189), c = logg_obs, s=50, vmin=min(logg_obs[logg_obs>0]), vmax=max(logg_obs), zorder=2)
pl.plot(temp_obs[a], 10**log_period_obs[a]/(10**(log_age_obs[a])**.5189), 'r.')
temp_plot = np.linspace(min(temp_obs), max(temp_obs))
pl.plot(temp_plot, (6250-temp_plot)**.2, 'r-', zorder=2)
pl.xlim(pl.gca().get_xlim()[::-1])
pl.xlabel('$\mathrm{T_{eff}~(K)}$')
pl.ylabel('$P_{rot}/A^n$')
pl.ylim(0, 15)
# pl.colorbar()
pl.savefig("init_teff2")

plots = pretty5.plotting()
plots.p_vs_t(par_true)

# plot ms turnoff
pl.clf()
pl.plot(temp_obs, logg_obs, 'k.')
coeffs = MS_poly()
# turnoff = np.polyval(coeffs, temp_obs)-.1
turnoff = np.polyval(coeffs, temp_obs)
pl.plot(temp_obs, turnoff, 'ro')
pl.ylim(pl.gca().get_ylim()[::-1])
pl.xlim(pl.gca().get_xlim()[::-1])
pl.savefig('t_vs_l')

raw_input('enter')

par_true = true_pars

# Now generate samples
nsamp = 100
log_age_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_age_obs, log_age_err)])
temp_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(temp_obs, temp_err)])
logg_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(logg_obs, logg_err)])
log_period_samp = np.vstack([x0+xe*np.random.randn(nsamp) for x0, xe in zip(log_period_obs, log_period_err)])
# FIXME: asymmetric errorbars for age and logg

# calculate ms turnoff coeffs
coeffs = MS_poly()

print 'initial likelihood = ', lnlike(par_true, log_age_samp, temp_samp, \
        log_period_samp, logg_samp, temp_obs, temp_err, log_period_obs, log_period_err, \
        logg_obs, logg_err, coeffs, Tk)

nwalkers, ndim = 32, len(par_true)
p0 = [par_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
args = (log_age_samp, temp_samp, log_period_samp, logg_samp, temp_obs, \
        temp_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, Tk)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

print("Burn-in")
p0, lp, state = sampler.run_mcmc(p0, 500)
sampler.reset()
print("Production run")
sampler.run_mcmc(p0, 4000)

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
fig_labels = ["$log(a)$", "$n$", "$b$", "$Y$", "$V$", "$Z$", "$U$", "$X$", "$W$"]
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