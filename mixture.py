# This script is the current working version. It was originally intended to be a mixture model, but now it's a composite.

import emcee
import triangle
import numpy as np
import matplotlib.pyplot as pl
from scipy.misc import logsumexp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.optimize as op

def plot(x, y, z, xerr, yerr, zerr, m):
    a = y > m[3]
    b = y < m[3]
#     a = y > 0.6
#     b = y < 0.6

    xs = np.linspace(min(x), max(x), num=500)
    ys = np.linspace(m[3], max(y), num=500)
    zs = model(m_true, xs, ys)

    # Load data
    data = np.genfromtxt('/Users/angusr/Python/Gyro/data/data.txt').T
    cols = np.genfromtxt("/Users/angusr/Python/Gyro/data/colours.txt")
    period = data[1]
    l = (period > 1.)
    period = np.log10(data[1][l])
    age = np.log10(data[13][l]*1000) # Convert to Myr
    bv = cols[1][l]

    pl.clf()
    pl.subplot(2,1,1)
    pl.errorbar(y[a], (10**z[a]), xerr = y_err[a], yerr = z_err[a], fmt = 'k.')
    pl.errorbar(y[b], (10**z[b]), xerr = y_err[b], yerr = z_err[b], fmt = 'r.')
    pl.plot(bv, 10**age, 'c.')
    pl.plot(ys, 10**zs, 'b-')
    pl.ylabel('age')
    pl.xlabel('colour')
#     zs = cmodel(m_true, ys)

    pl.subplot(2,1,2)
    pl.errorbar(10**z[a], (10**x[a]), xerr = z_err[a], yerr = x_err[a], fmt = 'k.')
    pl.errorbar(10**z[b], (10**x[b]), xerr = z_err[b], yerr = x_err[b], fmt = 'r.')
    pl.plot(10**age, 10**period, 'c.')
    pl.plot(10**zs, 10**xs, 'b-')
    pl.xlabel('age')
    pl.ylabel('period')
    pl.savefig("fakedata")

# generative model
def g_model(m, x, y): # model computes log(t) from log(p) and bv
    z = np.ones_like(y)
    a = y > m[3]
    b = y < m[3]
#     z[a] = 1./m[0] * ( x[a] - np.log10(m[1]) - \
#                 m[2]*np.log10(y[a] - m[3]))
    z[a] = 1./m[0] * (x[a] - np.log10(m[1]) - m[2]*np.log10(y[a]))
    z[b] = np.random.normal(3.5, 0.2, len(z[b]))
    return z

# Generate some fake data set
def fake_data(m_true, N):

    x = np.random.uniform(0.5, 1.8, N) # log(period)
    y = np.random.uniform(0.2, 1.,N) # colour
#     y = np.random.uniform(0.6, 1.2,N) # colour
    z = g_model(m_true, x, y) # log(age)

    # observational uncertainties.
    x_err = 0.01+0.01*np.random.rand(N)
    y_err = 0.01+0.01*np.random.rand(N)
    z_err = 0.01+0.01*np.random.rand(N)

    # add noise
    z_obs = z+z_err*np.random.randn(N)
    x_obs = x+x_err*np.random.randn(N)
    y_obs = y+y_err*np.random.randn(N)
    return x, y, z, x_obs, y_obs, z_obs, x_err, y_err, z_err

def model(m, x, y):
#     return 1./m[0]*(x - np.log10(m[1]) - m[2]*np.log10(y - m[3]))
    return 1./m[0]*(x - np.log10(m[1]) - m[2]*np.log10(y))

# Create fake data: n, a, b, c
m_true = [0.5189,  0.7725, 0.601, 0.4]
x, y, z, x_obs, y_obs, z_obs, x_err, y_err, z_err = fake_data(m_true, 100)

print "plotting data"
plot(x_obs, y_obs, z_obs, x_err, y_err, z_err, m_true)

# Draw posterior samples.
K = 500
x_samp = np.vstack([x0+xe*np.random.randn(K) for x0, xe in zip(x_obs, x_err)])
y_samp = np.vstack([y0+ye*np.random.randn(K) for y0, ye in zip(y_obs, y_err)])

# lhf
def lnlike(m):
    z_pred = model(m, x_samp, y_samp)
    chi2 = -0.5*((z_obs[:, None] - z_pred)/z_err[:, None])**2
    chi2[np.isnan(chi2)] = -np.inf
    return np.sum(np.logaddexp.reduce(chi2, axis=1))

# Flat priors
def lnprior(m):
    if np.any(m[:3] < 0.) == False  and np.any(1. < m) == False:
        return 0.0
    return -np.inf

def lnprob(m):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m)

print lnlike(m_true)
raw_input('enter')

# Sample the posterior probability for m.
nwalkers, ndim = 32, len(m_true)
p0 = [m_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
print("Burn-in")
p0, lp, state = sampler.run_mcmc(p0, 100)
sampler.reset()
print("Production run")
sampler.run_mcmc(p0, 500)

print("Making triangle plot")
fig_labels = ["$n$", "$a$", "$b$", "$c$", "$\mu_{age}$", "$\sigma_{age}$", "$P$"]
fig = triangle.corner(sampler.flatchain, truths=m_true, labels=fig_labels[:len(m_true)])
fig.savefig("triangle.png")

print("Plotting traces")
pl.figure()
for i in range(ndim):
    pl.clf()
    pl.plot(sampler.chain[:, :, i].T)
    pl.savefig("{0}.png".format(i))

# Flatten chain
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Find values
mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                  zip(*np.percentile(samples, [16, 50, 84], axis=0)))

print 'initial values', m_true
mcmc_result = np.array(mcmc_result)[:, 0]
print 'mcmc result', mcmc_result
