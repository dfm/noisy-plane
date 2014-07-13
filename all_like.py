import numpy as np
import matplotlib.pyplot as pl

# everywhere i've written temp, i mean bv.

def period_model(par, age, bv, c):
    print age[:5]
    return par[0] * (age*1e3)**par[1] * (bv-c)**par[2]

def age_model(par, period, bv, c):
#     age = ((period/(par[0]*(bv - c)**(par[2]))) ** 1./par[1])/1e3
#     print age[:5]
#     return ((1./par[0] * period * (bv - c)**(-par[2])) ** 1./par[1])/1e3
#     return ((period/(par[0]*(bv - c)**(par[2]))) ** 1./par[1])/1e3
    return np.exp((1./par[1])*(np.log(period)-np.log(par[0])-par[2]*(bv-c)))/1000.

def lnlike(par, age_samp, temp_samp, period_samp, logg_samp, age_obs, age_err, \
               temp_obs, temp_err, period_obs, period_err, logg_obs, logg_err, coeffs, c):
    nobs, nsamp = age_samp.shape

#     pl.clf()
#     pl.errorbar(age_obs, period_obs, xerr=age_err, capsize=0, fmt='k.')
#     ys = np.linspace(0, max(period_obs), 100)
#     xs = np.linspace(0, max(age_obs), 100)
#     pl.plot(age_model(par, ys, 0.65, c), ys)
#     pl.plot(xs, period_model(par, xs, 0.65, c))
#     pl.savefig('test')
#     raw_input('enter')

    age_pred = age_model(par[:3], period_samp, temp_samp, c)
    Y, V = par[3], par[4]
    Z, W = par[5], par[6]
    X, U, P = par[7], par[8], par[9]
    logg_cut = 4.

    ll = np.zeros(nobs)
    for i in np.arange(nobs):

        # cool MS stars
        l1 = (temp_samp[i,:] > c) * (logg_samp[i,:] > logg_cut)
        if l1.sum() > 0:
            loglike11 = -.5*((age_obs[i]-age_pred[i,l1])**2/age_err[i]**2)
            loglike12 = -.5*((age_obs[i]-X)**2/(U+age_err[i]**2))
            loglike1 = loglike11 + np.log((1-P)/age_err[i]) \
                    + loglike12 + np.log(P/np.sqrt(U+age_err[i]**2))
            lik1 = np.logaddexp.reduce(loglike1, axis=0) / float(l1.sum())
        else:
            lik1 = 0.

        # hot MS stars
        l2 = (temp_samp[i,:] < c) * (logg_samp[i,:] > logg_cut)
        if l2.sum() > 0:
            loglike2 = -.5*((age_obs[i]-Y)**2/(V+age_err[i]**2))
            lik2 = np.logaddexp.reduce(loglike2, axis=0) / float(l2.sum())
        else:
            lik2 = 0.

        # subgiants
        l3 = (logg_samp[i,:] < logg_cut)
        if l3.sum() > 0:
            loglike3 = -.5*((age_obs[i]-Z)**2/(W+age_err[i]**2))
            lik3 = np.logaddexp.reduce(loglike3, axis=0) / float(l2.sum())
        else:
            lik3 = 0.
        lik12 = np.logaddexp(lik1, lik2)
        ll[i] = np.logaddexp(lik12, lik3)
    ll[np.isnan(ll)] = -np.inf
    return np.logaddexp.reduce(ll, axis=0)
