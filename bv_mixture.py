# using colours instead of teffs...
import emcee
import triangle
import numpy as np
import matplotlib.pyplot as pl
from scipy.misc import logsumexp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.optimize as op
import plotting
import models
from load_hyades import hya_load
from mpl_toolkits.mplot3d import Axes3D

# Suzanne's lhf
def lnlike(par):
    par = np.exp(par)
    C_MAX = 0.2
    C_MIN = 1.3
    z_pred = 10**(models.ibc_model(par, np.log10(x_samp), y_samp))
#     print z_pred
    nobs,nsamp = x_samp.shape
    ll = np.zeros(nobs)
    temp_Kraft = .4
    A = max(0,(C_MAX- temp_Kraft) / float(C_MAX - C_MIN))
    ll = np.zeros(nobs)
    for i in np.arange(nobs):
        l1 = y_samp[i,:] > temp_Kraft
        if l1.sum() > 0:
            like1 = -0.5*((z_obs[i] - z_pred[i][l1])/z_err[i])**2 - np.log(z_err[i])
            lik1 = np.logaddexp.reduce(like1, axis=0) -np.log(float(l1.sum()))
        else:
            lik1 = -np.inf
        l2 = l1 == False
        if l2.sum() > 0:
            like2 = np.log(A)*(-((y_obs[i] - y_samp[i][l2])/2.0/y_err[i])**2) -np.log(y_err[i])
            lik2 = np.logaddexp.reduce(like2, axis=0) -np.log(float(l2.sum()))
        else:
            lik2 = -np.inf
        ll[i] = np.logaddexp(lik1, lik2)
#     if np.any(np.isnan(ll)):
#         print 'll', ll
#         print 'l1', lik1
#         print 'l2', lik2
    ll[np.isnan(ll)] = -np.inf
#     print np.logaddexp.reduce(ll, axis=0)
    return np.logaddexp.reduce(ll, axis=0)

# # Suzanne's lhf
# def lnlike(m):
#     m = np.exp(m)
#     C_MAX = .2
#     C_MIN = 1.3
#     nobs,nsamp = x_samp.shape
#     z_pred = 10**models.ibc_model(m, np.log10(x_samp), y_samp)
#     ll = np.zeros(nobs)
# #     temp_Kraft = m[3]
#     temp_Kraft = .4
#     A = max(0,(C_MAX- temp_Kraft) / float(C_MAX - C_MIN))
#     ll = np.zeros(nobs)
#     for i in np.arange(nobs):
#         l1 = y_samp[i,:] < temp_Kraft
#         if l1.sum() > 0:
#             like1 = \
#                 np.exp(-((z_obs[i] - z_pred[i,l1])/2.0/z_err[i])**2) \
#                 / z_err[i]
#             lik1 = np.sum(like1) / float(l1.sum())
#         else:
#             lik1 = 0.0
#         l2 = l1 == False
#         if l2.sum() > 0:
#             like2 = A * \
#                 np.exp(-((y_obs[i] - y_samp[i,l2])/2.0/y_err[i])**2) \
#                 / y_err[i]
#             lik2 = np.sum(like2) / float(l2.sum())
#         else:
#             lik2 = 0.0
#         ll[i] = np.log10(lik1 + lik2)
#     ll[np.isinf(ll)] = 0.
#     return np.sum(ll)

# uniform Priors (for 4 parameter model)
def lnprior(m):
#     m = np.exp(m)
#     if -10.<m[0]<10. and -10.<m[1]<10. and -10.<m[2]<10. and -10.<m[3]<10.:
    if -10.<m[0]<10. and -10.<m[1]<10. and -10.<m[2]<10.:
        return 0.0
    return -np.inf

# posterior
def lnprob(m):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m)

if __name__ == "__main__":

#     m_true = [0.5189, 0.7725, 0.601, 0.4]
    m_true = [0.5189, 0.7725, 0.601]
    m_true = np.log(m_true)

    # loading hyades + sun data
    x_obs, y_obs, z_obs, x_err, y_err, z_err = hya_load()

    pl.clf()
#     y_plot = np.linspace(min(y_obs), max(y_obs), 100)
    y_plot = np.linspace(.6, max(y_obs), 100)
    pl.errorbar(y_obs[1:], x_obs[1:], xerr=y_err[1:], yerr=x_err[1:], fmt = 'k.')
    pl.errorbar(y_obs[0], x_obs[0], xerr=y_err[0], yerr=x_err[0], fmt = 'r.')
    pl.plot(y_plot, 10**(models.bc_model(m_true, np.log10(1000), y_plot)), 'r-')
    pl.plot(y_plot, 10**(models.bc_model(m_true, np.log10(650), y_plot)), 'b-')
    pl.plot(y_plot, 10**(models.bc_model(m_true, np.log10(800), y_plot)), 'm-')
    pl.plot(y_plot, 10**(models.bc_model(m_true, np.log10(400), y_plot)), 'c-')
    pl.savefig('hyades')

    pl.clf()
    x_plot = np.linspace(min(x_obs), max(x_obs), 100)
    pl.errorbar(x_obs[1:], z_obs[1:], xerr=x_err[1:], yerr=z_err[1:], fmt = 'k.')
    pl.errorbar(x_obs[0], z_obs[0], xerr=x_err[0], yerr=z_err[0], fmt = 'r.')
    pl.plot(x_plot, 10**(models.ibc_model(m_true, np.log10(x_plot), .6)), 'r-')
    pl.plot(x_plot, 10**(models.ibc_model(m_true, np.log10(x_plot), .7)), 'b-')
    pl.plot(x_plot, 10**(models.ibc_model(m_true, np.log10(x_plot), 1.)), 'm-')
    pl.plot(x_plot, 10**(models.ibc_model(m_true, np.log10(x_plot), 1.1)), 'c-')
    pl.savefig('hyades2')

    pl.clf()
    fig = pl.figure(1)
    ax = fig.gca(projection='3d')
    ax.scatter(x_obs, y_obs, z_obs, c = 'b', marker = 'o')
    x_surf=np.r_[min(x_obs):max(x_obs):100j]
    y_surf=np.r_[min(y_obs):max(y_obs):100j]
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = 10**(models.ibc_model(m_true, np.log10(x_surf), y_surf))
    ax.plot_surface(x_surf, y_surf, z_surf, alpha = 0.2)
    ax.set_xlabel('Rotational period (days)')
    ax.set_ylabel('B-V')
    ax.set_zlabel('Age (Gyr)')
    pl.savefig('bv_3d')

    print "Draw posterior samples."
    K = 500
    x_samp = np.vstack([x0+xe*np.random.randn(K) for x0, xe in zip(x_obs, x_err)])
    y_samp = np.vstack([y0+ye*np.random.randn(K) for y0, ye in zip(y_obs, y_err)])

    print 'initial likelihood = ', lnlike(m_true)
    print 'initial likelihood = ', lnlike([0.6, 0.7725, 0.601])
    raw_input('enter')

    # Sample the posterior probability for m.
    nwalkers, ndim = 32, len(m_true)
    p0 = [m_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, 500)
    sampler.reset()
    print("Production run")
    sampler.run_mcmc(p0, 2000)

    print("Making triangle plot")
    fig_labels = ["$n$", "$a$", "$b$", "$T_K$"]
    fig = triangle.corner(sampler.flatchain, truths=m_true, labels=fig_labels[:len(m_true)])
    fig.savefig("triangle.png")

    print("Plotting traces")
    pl.figure()
    for i in range(ndim):
        pl.clf()
        pl.axhline(m_true[i], color = "r")
        pl.plot(sampler.chain[:, :, i].T, 'k-', alpha=0.3)
        pl.savefig("{0}.png".format(i))

    # Flatten chain
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    # Find values
    mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(samples, [16, 50, 84], axis=0)))

    print 'initial values', m_true
    mcmc_result = np.array(mcmc_result)[:, 0]
    # print "ml result = ", result
    print 'mcmc result', mcmc_result
