import emcee
import triangle
import numpy as np
import matplotlib.pyplot as pl
from scipy.misc import logsumexp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# def model(m, x, y): # now model computes log(t) from log(p) and bv
#     return 1./m[0] * ( x - np.log10(m[1]) - m[2]*np.log10(y - m[3]))

# def model_data(m, x, y): # now model computes log(t) from log(p) and bv
#     z = np.ones_like(x)
#     for i in range(len(y)):
#         if y[i] > k_break:
#             z[i] = 1./m[0]*(x[i] - np.log10(m[1]) - \
#                     m[2]*np.log10(y[i] - m[3]))
#         elif y[i] < k_break:
#             z[i] = np.random.uniform(2., 4.)
#     return z

# def model(m, x, y): # now model computes log(t) from log(p) and bv
#     z = np.ones_like(x)
#     for i in range(len(y)):
#         for j in range(len(y[i])):
#             if y[i][j] > k_break: z[i][j] = 1./m[0] * ( x[i][j] - np.log10(m[1]) - \
#                         m[2]*np.log10(y[i][j] - m[3]))
#             elif y[i][j] < k_break:
#                 z[i][j] = np.random.uniform(2., 8.)
#     return z

def model(m, x, y): # now model computes log(t) from log(p) and bv
    z = np.ones_like(y)
    a = y > k_break
    b = y < k_break
    z[a] = 1./m[0] * ( x[a] - np.log10(m[1]) - \
                m[2]*np.log10(y[a] - m[3]))
     z[b] = np.random.uniform(2., 4., len(z[b]))
#     z[b] = np.random.uniform(2., 4., len(z[b]))
    return z

# def model(m, x, y): # now model computes log(t) from log(p) and bv
#     return 1./m[0]*(x - np.log10(m[1]) -  m[2]*np.log10(y - m[3]))

# def model(m, x, y): # model computes log(t) from log(p) and teff
#     return (x - m[0]*np.log10(y - m[1]))/(m[2]*(y - m[1])) + m[3]

# Generate true values.
N = 50
k_break = 0.6
m_true = [0.5189,  0.7725, 0.601, 0.4]
# m_true = [0.5189,  0.7725, 0.601, 0.4, 3.5, 0.5] # Added extra params for mean and variance of age dist.
x = np.random.uniform(0.5, 1.8, N) # log period
y = np.random.uniform(0.2, 1.2,N)
z = model(m_true, x, y) #age
print 10**z[:5], 'age'
print min(z), max(z)
print 10**x[:5], 'period'
print y[:5], 'color'

# observational uncertainties.
x_err = 0.01+0.01*np.random.rand(N)
y_err = 0.01+0.01*np.random.rand(N)
z_err = 0.01+0.01*np.random.rand(N)

z_obs = z+z_err*np.random.randn(N)
x_obs = x+x_err*np.random.randn(N)
y_obs = y+y_err*np.random.randn(N)

# plot fake data
pl.clf()
a = y > k_break
b = y < k_break
pl.subplot(2,1,1)
xs = np.linspace(min(x), max(x), num=500)
ys = np.linspace(min(y), max(y), num=500)
zs = model(m_true, xs, ys)
pl.errorbar(y[a], (10**z[a]), xerr = y_err[a], yerr = 10**z_err[a], fmt = 'k.')
pl.errorbar(y[b], (10**z[b]), xerr = y_err[b], yerr = 10**z_err[b], fmt = 'r.')
pl.plot(ys, zs, 'b-')
pl.ylabel('age')
pl.xlabel('colour')
pl.subplot(2,1,2)
pl.errorbar(10**x[a], (10**z[a]), xerr = x_err[a], yerr = z_err[a], fmt = 'k.')
pl.errorbar(10**x[b], (10**z[b]), xerr = x_err[b], yerr = z_err[b], fmt = 'r.')
pl.plot(xs, zs, 'b-')
pl.ylabel('age')
pl.xlabel('period')
pl.savefig("fakedata")

# pl.close(1)
# fig = pl.figure(1)
# ax = fig.gca(projection = '3d')
# ax.set_xlabel('Period')
# ax.set_ylabel('Teff')
# ax.set_zlabel('Age')
# x_surf = np.arange(min(x), max(x), 0.01)                # generate a mesh
# y_surf = np.arange(min(y), max(y), 0.01)
# x_surf, y_surf = np.meshgrid(x_surf, y_surf)
# z_surf = model(m_true, x_surf, y_surf)
# ax.plot_surface(10**x_surf, y_surf, 10**z_surf, cmap = cm.hot, alpha = 0.2);    # plot a 3d surface plot

# # Load data
# data = np.genfromtxt('/Users/angusr/Python/Gyro/data/matched_data.txt').T
# period = data[1]

# # remove stars with period < 1
# a = (period > 1.)

# # Assign variable names
# x = data[1][a]
# xerrp = data[2][a]
# xerrm = data[2][a]
# z = data[3][a]*1000 # Convert to Myr
# zerrp = data[4][a]*1000
# zerrm = data[5][a]*1000

# # make up colours
# y = np.random.uniform(k_break,1.2,len(z))
# yerr = np.ones_like(y) * 0.05

# print z[:5], 'age'
# print x[:5], 'period'
# print y[:5], 'color'

# pl.close(2)
# fig = pl.figure(2)
# ax = fig.gca(projection = '3d')
# ax.scatter(x, y, z, color = 'b')
# ax.set_xlabel('Period')
# ax.set_ylabel('Teff')
# ax.set_zlabel('Age')
# pl.show()

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

def lnprior(m):
    if np.any(m < 0.) == False and np.any(1. < m) == False:
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
