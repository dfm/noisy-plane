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
import models
from load_hyades import hya_load

# Suzanne's lhf
def lnlike(par):
    TEMP_MAX = 7500
    TEMP_MIN = 3000
    z_pred = models.model(par, x_samp, y_samp)
    nobs,nsamp = x_samp.shape
    ll = np.zeros(nobs)
    temp_Kraft = 6500.
#     temp_Kraft = par[3]
    A = max(0,(TEMP_MAX- temp_Kraft) / float(TEMP_MAX - TEMP_MIN))
    ll = np.zeros(nobs)
    for i in np.arange(nobs):
        l1 = y_samp[i,:] < temp_Kraft
        if l1.sum() > 0:
            like1 = \
                np.exp(-((np.log10(z_obs[i]) - z_pred[i][l1])/2.0/z_err[i])**2) \
                / z_err[i]
            lik1 = np.sum(like1) / float(l1.sum())
        else:
            lik1 = 0.0
        l2 = l1 == False
        if l2.sum() > 0:
            like2 = A * \
                np.exp(-((y_obs[i] - y_samp[i][l2])/2.0/y_err[i])**2) \
                / y_err[i]
            lik2 = np.sum(like2) / float(l2.sum())
        else:
            lik2 = 0.0
        ll[i] = np.log10(lik1 + lik2)
    return np.sum(ll)

# uniform Priors (for 4 parameter model)
def lnprior(m):
    if -10. < m[0] < 10. and -10. < m[1] < 10. and -10. < m[2] < 10.:
        return 0.0
    return -np.inf

# posterior
def lnprob(m):
    lp = lnprior(m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(m)

# print "Calculating maximum-likelihood values"
# print "initial likelihood", lnlike(m_true)
# args = None
# nll = lambda *args: -lnlike(*args)
# # nll = lambda: -lnlike
# result = op.fmin(-lnlike, m_true, args = args)
# # print "final likelihood", lnlike(result, x_samp, y_samp, y_obs, y_err, x_obs, x_err)
# print "final likelihood", lnlike(result)
# print "ml params = ", result
# # plotting.plt(x_obs, y_obs, z_obs, x_err, y_err, z_err, result, "ml_result")
# # plotting.plot3d(x_obs, y_obs, z_obs, x_obs, y_obs, z_obs, result, 2, 'b', "3dml")

if __name__ == "__main__":
    # m_true = [1.927, 0.216, 0.1156, 6500.] # latest version
    m_true = [1.927, 0.216, 0.1156] # fixed T_k

    # generating fake data
    # x_obs, y_obs, z_obs, x_err, y_err, z_err = plotting.fake_data(m_true, 144)
    # plotting.plt(x_obs, y_obs, z_obs, x_err, y_err, z_err, m_true, "fakedata")

    # loading real data
    x_obs, y_obs, z_obs, x_err, y_err, z_err = plotting.load_dat()
    # plotting.plt(x_obs, y_obs, z_obs, x_err, y_err, z_err, m_true, "realdata")

#     # loading hyades + sun data
#     x_obs, y_obs, z_obs, x_err, y_err, z_err = hya_load()
#     pl.clf()
#     pl.errorbar(y_obs[1:], x_obs[1:], xerr=y_err[1:], yerr=x_err[1:], fmt = 'k.')
#     pl.errorbar(y_obs[0], x_obs[0], xerr=y_err[0], yerr=x_err[0], fmt = 'r.')
#     pl.savefig('hyades')
#     raw_input('enter')

    print '3d plot'
    # 3d plot
    # a = y_obs < m_true[3]
    # plotting.plot2d(x_obs[a], y_obs[a], z_obs[a], x_obs[a], y_obs[a], z_obs[a], m_true, 1, 'k', "3dorig")
    plotting.plot3d(x_obs, y_obs, z_obs, x_obs, y_obs, z_obs, m_true, 1, 'k', "3dorig")
    raw_input('enter')

    print "Draw posterior samples."
    K = 500
    x_samp = np.vstack([x0+xe*np.random.randn(K) for x0, xe in zip(x_obs, x_err)])
    y_samp = np.vstack([y0+ye*np.random.randn(K) for y0, ye in zip(y_obs, y_err)])

    # Sample the posterior probability for m.
    nwalkers, ndim = 32, len(m_true)
    p0 = [m_true+1e-4*np.random.rand(ndim) for i in range(nwalkers)]
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    print("Burn-in")
    p0, lp, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    print("Production run")
    sampler.run_mcmc(p0, 500)

    print("Making triangle plot")
    fig_labels = ["$n$", "$a$", "$b$", "$T_K$", "$Z$", "$V$"]
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

    # plotting result
    # plotting.plt(x_obs, y_obs, z_obs, x_err, y_err, z_err, mcmc_result, "mcmc_result")
    # plotting.plot3d(x_obs, y_obs, z_obs, x_obs, y_obs, z_obs, mcmc_result, 3, 'r', "3dmcmc")
