# Version with fake data
import emcee
import triangle
import numpy as np
import matplotlib.pyplot as pl
from scipy.misc import logsumexp

# def model(m, x, y):
#     n, a, b, c = m
#     return n * x + np.log10(a) + b*np.log10(y - c)

def model(m, x, y): # now model computes log(t) from log(p) and bv
    return 1./m[0] * ( x - np.log10(m[1]) - m[2]*np.log10(y - m[3]))

# def model(m, x, y):
#     return m[0]+m[1]*x+m[2]*y

# Generate true values.
N = 50
# m_true = [0.5, 0.6, 0.1]
m_true = [0.5189,  0.7725, 0.601, 0.4]
x = 1 + 2*np.random.rand(N)
y = 0.4+np.random.rand(N)
z = model(m_true, x, y)

# observational uncertainties.
x_err = 0.01+0.01*np.random.rand(N)
y_err = 0.01+0.01*np.random.rand(N)
z_err = 0.01+0.05*np.random.rand(N)

z_obs = z+z_err*np.random.randn(N)
x_obs = x+x_err*np.random.randn(N)
y_obs = y+y_err*np.random.randn(N)

# # Switch x and z
# x2 = x; xerr2 = xerr
# x = z; xerr = zerr
# z = x2; zerr = xerr2

# Draw posterior samples.
K = 500
x_samp = np.vstack([x0+xe*np.random.randn(K) for x0, xe in zip(x_obs, x_err)])
y_samp = np.vstack([y0+ye*np.random.randn(K) for y0, ye in zip(y_obs, y_err)])

# Dan's original lhf
def lnlike(m):
    z_pred = model(m, x_samp, y_samp)
    chi2 = -0.5*((z_obs[:, None] - z_pred)/z_err[:, None])**2
    chi2[np.isnan(chi2)] = -np.inf
    return np.sum(np.logaddexp.reduce(chi2, axis=1))

# # Suzanne's lhf
# def lnlike(m):
#     sr = 1.0/(z_err[:, None]**2) * (z[:, None]-model(m, x_samp, y_samp))**2
#     N = np.array(((np.isfinite(sr)).sum(axis = 1)), dtype = float)
#     z_err[np.isnan(z_err)] = 0.
#     sr[np.isnan(sr)] = 0.
#     print - 0.5 * float(N) * np.log(2 * np.pi)
#     print - sum(np.log(z_err[:, None])), np.shape(sum(np.log(z_err[:, None])))
#     print - 0.5 * np.logaddexp.reduce(sr, axis = 1), np.shape(0.5 * np.logaddexp.reduce(sr, axis = 1))
#     logL = - 0.5 * float(N) * np.log(2 * np.pi) \
#       - sum(np.log(z_err[:, None])) \
#       - 0.5 * sum(sr)
#     print logL 
#     return logL
 
# def lnlike(m):
#     z_pred = model(m, x_samp, y_samp)
#     sr = 1.0/(z_err[:, None]**2) * (z[:, None]-z_pred)**2
#     N = np.array(((np.isfinite(sr)).sum(axis = 1)), dtype = float)
#     #FIXME: will N ever be less than 50? if not, don't worry!
# #     raw_input('enter')
#     chi2 = -0.5*((z[:, None] - z_pred)/z_err[:, None])**2
#     chi2[np.isnan(chi2)] = 0.
# #     chi2[i,:] = [chi2[i,:][np.isfinite(chi2[i,:])] for i in range(len(x_samp))] # remove NaNs
# #     raw_input('enter')
# #     print - 0.5 * len(N) * np.log(2*np.pi) - sum(np.log(z_err[:, None]))\
# #         - 0.5 * np.sum(np.logaddexp.reduce(chi2, axis=1))
# 
#     return - 0.5 * len(N) * np.log(2*np.pi) - sum(np.log(z_err[:, None]))\
#         - 0.5 * np.sum(np.logaddexp.reduce(chi2, axis=1))
 
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
nwalkers, ndim = 32, len(m_true)
p0 = [m_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
print("Burn-in")
p0, lp, state = sampler.run_mcmc(p0, 100)
sampler.reset()
print("Production run")
sampler.run_mcmc(p0, 500)

print("Making triangle plot")
fig = triangle.corner(sampler.flatchain, truths=m_true,
                      labels=["$n$", "$a$", "$b$", "$c$"])
# fig = triangle.corner(sampler.flatchain, truths=m_true,
#                       labels=["$m_0$", "$m_1$", "$m_2$"])
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
print 'initial values', m_true
mcmc_result = np.array(mcmc_result)[:,0]
print 'mcmc result', mcmc_result
