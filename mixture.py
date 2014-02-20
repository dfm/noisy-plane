# This script is the current working version. It was originally intended to be a mixture model, but now it's a composite.

import emcee
import triangle
import numpy as np
import matplotlib.pyplot as pl
from scipy.misc import logsumexp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.optimize as op
import plotting

def model(m, x, y):
    return 1./m[0]*(x - np.log10(m[1]) - m[2]*np.log10(y))

print "loading real data"
xr, yr, zr, xer, yer, zer = plotting.load()

print "generating fake data"
m_true = [0.5189,  0.7725, 0.601]
x, y, z, x_obs, y_obs, z_obs, x_err, y_err, z_err = plotting.fake_data(m_true, 144)

# replacing values slowly
rp = 140
x_obs[:rp] = xr[:rp]
y_obs[:rp] = yr[:rp]
z_obs[:rp] = zr[:rp]
x_err[:rp] = xer[:rp]
y_err[:rp] = yer[:rp]
z_err[:rp] = zer[:rp]

print 10**(z_obs[:2*rp])
print (max(10**(z_obs[:2*rp]))), "Gyr"

# print "loading real data"
# x_obs, y_obs, z_obs, x_err, y_err, z_err = plotting.load()

print "plotting data"
plotting.plt(x_obs, y_obs, z_obs, x_err, y_err, z_err, m_true, "fakedata")
plotting.plot3d(x_obs, y_obs, z_obs, x_obs, y_obs, z_obs, m_true, 1)

# Draw posterior samples.
K = 500
x_samp = np.vstack([x0+xe*np.random.randn(K) for x0, xe in zip(x_obs, x_err)])
y_samp = np.vstack([y0+ye*np.random.randn(K) for y0, ye in zip(y_obs, y_err)])

print "no. colours < 0:", sum(sum(y_samp<0))

# lhf
def lnlike(m):
    z_pred = model(m, x_samp, y_samp)
    chi2 = -0.5*((z_obs[:, None] - z_pred)/z_err[:, None])**2
    chi2[np.isnan(chi2)] = -np.inf
    return np.sum(np.logaddexp.reduce(chi2, axis=1))

# Gaussian priors
def lnprior(m):
    return -0.5*(m[0]+0.5)**2 -0.5*(m[1]+0.5)**2 -0.5*(m[2]+0.5)**2
#     return -0.5*(m[0]+1.)**2 -0.5*(m[1]+1.)**2 -0.5*(m[2]+1.)**2

# posterior
def lnprob(m):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m)

print "Calculating maximum-likelihood values"
nll = lambda *args: -lnlike(*args)
result = op.fmin(nll, m_true)
print "ml result = ", result
plotting.plt(x_obs, y_obs, z_obs, x_err, y_err, z_err, result, "ml_result")
plotting.plot3d(x_obs, y_obs, z_obs, x_obs, y_obs, z_obs, result, 2)

print "lnlike = ", lnlike(m_true)
raw_input('enter')

# Sample the posterior probability for m.
nwalkers, ndim = 32, len(m_true)
p0 = [m_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
print("Burn-in")
p0, lp, state = sampler.run_mcmc(p0, 100)
sampler.reset()
print("Production run")
sampler.run_mcmc(p0, 700)

print("Making triangle plot")
fig_labels = ["$n$", "$a$", "$b$", "$c$", "$\mu_{age}$", "$\sigma_{age}$", "$P$"]
fig = triangle.corner(sampler.flatchain, truths=m_true, labels=fig_labels[:len(m_true)])
fig.savefig("triangle.png")

print("Plotting traces")
pl.figure()
for i in range(ndim):
    pl.clf()
    pl.plot(m_true[i], "k-")
    pl.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3)
    pl.savefig("{0}.png".format(i))

# Flatten chain
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Find values
mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                  zip(*np.percentile(samples, [16, 50, 84], axis=0)))

print 'initial values', m_true
mcmc_result = np.array(mcmc_result)[:, 0]
print 'mcmc result', mcmc_result

# plotting result
plotting.plt(x_obs, y_obs, z_obs, x_err, y_err, z_err, mcmc_result, "mcmc_result")
plotting.plot3d(x_obs, y_obs, z_obs, x_obs, y_obs, z_obs, mcmc_result, 3)
