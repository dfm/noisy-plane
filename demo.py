import emcee
import triangle
import numpy as np
import matplotlib.pyplot as pl
from scipy.misc import logsumexp

# Generate true values.
N = 50
m_true = [0.5, 1.0, 0.1]
x = 1 + 4*np.random.rand(N)
y = 10 + 40*np.random.rand(N)
z = m_true[0]+m_true[1]*x+m_true[2]*y

# Observational uncertainties.
x_err = 0.01+0.01*np.random.rand(N)
y_err = 1.0+1.0*np.random.rand(N)
z_err = 0.01+0.05*np.random.rand(N)

z_obs = z+z_err*np.random.randn(N)
x_obs = x+x_err*np.random.randn(N)
y_obs = y+y_err*np.random.randn(N)

# Draw posterior samples.
K = 500
x_samp = np.vstack([x0+xe*np.random.randn(K) for x0, xe in zip(x_obs, x_err)])
y_samp = np.vstack([y0+ye*np.random.randn(K) for y0, ye in zip(y_obs, y_err)])

# Define the marginalized likelihood function.
def lnlike(m):
    z_pred = m[0]+m[1]*x_samp+m[2]*y_samp
    chi2 = -0.5*((z_obs[:, None] - z_pred)/z_err[:, None])**2
    return np.sum(logsumexp(chi2))

def lnprob(m):
    return lnlike(m)

# Sample the posterior probability for m.
nwalkers, ndim = 32, len(m_true)
p0 = [m_true+1e-2*np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
print("Burn-in")
p0, lp, state = sampler.run_mcmc(p0, 100)
sampler.reset()
print("Production run")
sampler.run_mcmc(p0, 500)

print("Making triangle plot")
fig = triangle.corner(sampler.flatchain, truths=m_true,
                      labels=["$m_0$", "$m_1$", "$m_2$"])
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
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print 'true values', m_true
print 'mcmc result' = np.array(mcmc_result)[:,0]