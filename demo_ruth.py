# get working with noisy data
# get working with real data
# Change the model and change z to age, x to period and y to colour
import emcee
import triangle
import numpy as np
import matplotlib.pyplot as pl
from scipy.misc import logsumexp

def model(m, x, y):
    return m[0] * x + np.log10(m[1]) + m[2]*np.log10(y - m[3])

# def model(m, x, y): # now model computes log(t) from log(p) and bv
#     return 1./m[0] * ( x - np.log10(m[1]) - m[2]*np.log10(y - m[3]))

def log_errorbar(y, errp, errm):
    plus = y + errp
    minus = y - errm
    log_err = np.log10(plus/minus) / 2.
    l = minus < 0 # Make sure ages don't go below zero!
    log_err[l] = np.log10(plus[l]/y[l])
    return log_err

# Load data
data = np.genfromtxt('/Users/angusr/Python/Gyro/data/matched_data.txt').T
mass = data[6]
logg = data[9]
period = data[1]

# Remove subgiants, Kraft break stars and stars with period < 1
a = (period > 1.) * (logg > 4.) * (mass < 1.3)

# Assign variable names
x = data[1][a]
xerrp = data[2][a]
xerrm = data[2][a]
z = data[3][a]*1000 # Convert to Myr
zerrp = data[4][a]*1000
zerrm = data[5][a]*1000

# make up colours
y = np.random.uniform(0.4,1.2,len(z))
yerr = np.ones_like(y) * 0.05
l = y < 0.4

# Take logs
x = np.log10(x)
z = np.log10(z) # remove this line if using fake data

m_true = [0.5189,  0.7725, 0.601, 0.4]
# z = model(m_true, x, y) #+ np.random.randn(len(age)) # Fake z data
x = model(m_true, z, y) #+ np.random.randn(len(age)) # Fake x data

# Calculate logarithmic errorbars
zerr = log_errorbar(z, zerrp, zerrm)
xerr = log_errorbar(x, xerrp, xerrm)

# # Make up uncertainties for now
N = len(z)
zerr = 0.01+0.01*np.random.rand(N)
yerr = 0.01+0.01*np.random.rand(N)
xerr = 0.01+0.01*np.random.rand(N)

# Resample those points that are less than 0.4
while l.sum() > 0:
    y[l] = np.random.uniform(0.4,1.2,l.sum())
    yerr[l] = np.ones_like(y[l]) * 0.005
    l = y < 0.4

# switching x and z - necessary if using the first model
x2 = x; xerr2 = xerr
x = z; xerr = zerr
z = x2; zerr = xerr2

print 10**z[:5], 't'
print y[:5], 'B-V'
print 10**x[:5], 'P'
print zerr[:5], 't_err'
print yerr[:5], "bv_err"
print xerr[:5], "P_err"

raw_input('enter')

# Draw posterior samples.
K = 500
x_samp = np.vstack([x0+xe*np.random.randn(K) for x0, xe in zip(x, xerr)])
y_samp = np.vstack([y0+ye*np.random.randn(K) for y0, ye in zip(y, yerr)])

# Define the marginalized likelihood function.
def lnlike(m):
    z_pred = model(m, x_samp, y_samp)
    sr = 1.0/(zerr[:, None]**2) * (z[:, None]-z_pred)**2
    N = np.array(((np.isfinite(sr)).sum(axis = 1)), dtype = float)
    N[N==0.] = 1. 
    chi2 = -0.5*((z[:, None] - z_pred)/zerr[:, None])**2
    chi2[np.isnan(chi2)] = 0.
    return np.sum(np.logaddexp.reduce(chi2, axis=1)/N)

def lnprior(m):
    if np.any(m<0.)==False and np.any(1.<m)==False:
        return 0.0
    return -np.inf
        
def lnprob(m):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m)

# Sample the posterior probability for m.
nwalkers, ndim = 100, len(m_true)
p0 = [m_true+1e-2*np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
print("Burn-in")
p0, lp, state = sampler.run_mcmc(p0, 100)
sampler.reset()
print("Production run")
sampler.run_mcmc(p0, 500)

print("Making triangle plot")
fig = triangle.corner(sampler.flatchain, truths=m_true,
                      labels=["$n$", "$a$", "$b$", "$c$"])
fig.savefig("triangle.png")

print("Plotting traces")
pl.figure()
labels = ['n', 'a', 'b', 'c']
for j in range(ndim):
    pl.subplot(ndim,1,j)
    [pl.plot(sampler.chain[i, :, j], 'k-', \
        alpha = 0.2) for i in range(nwalkers)]
    pl.axhline(m_true[j], color = 'r')
    pl.ylabel('%s' %labels[j])
pl.savefig('traces')

# Flatten chain
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Find values
mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print 'initial values', m_true
mcmc_result = np.array(mcmc_result)[:,0]
print 'mcmc result', mcmc_result
