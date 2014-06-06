import numpy as np
import matplotlib.pyplot as pl

# everywhere i've written temp, i mean bv.

def period_model(par, age, bv, c):
    return par[0] * (age*1e3)**par[1] * (bv-c)**par[2]

def lnlike(par, age_samp, temp_samp, period_samp, logg_samp, \
               temp_obs, temp_err, period_obs, period_err, logg_obs, logg_err, coeffs, c):
    nobs, nsamp = age_samp.shape
    period_pred = period_model(par[:3], age_samp, temp_samp, c)
    Y, V = par[3], par[4]
    Z, U = par[5], par[6]
    X, W, P = par[7], par[8], par[9]
    logg_cut = 3.7 # FIXME

    ll = np.zeros(nobs)
    for i in np.arange(nobs):

        # cool MS stars
        l1 = (temp_samp[i,:] > c) * (logg_samp[i,:] > logg_cut)
        if l1.sum() > 0:
            like11 = \
                np.exp(-((period_obs[i] - period_pred[i,l1])/2.0/period_err[i])**2) \
                / period_err[i]
            like12 = \
                np.exp(-((period_obs[i] - X)**2/(2.0)**2/(W + period_err[i])**2)) \
                / (W + period_err[i]) # incorrect but works FIXME
            like1 = (1-P)*like11 + P*like12 # FIXME: not sure if this line is correct
            if i == nobs-1:
                period_err[i] = 1.
                like1 = np.exp(-((period_obs[i] - period_pred[i,l1])/2.0/period_err[i])**2) \
                / period_err[i]
            lik1 = np.sum(like1) / float(l1.sum())
        else:
            lik1 = 0.0

        # hot MS stars
        l2 = (temp_samp[i,:] < c) * (logg_samp[i,:] > logg_cut)
        if l2.sum() > 0:
            like2 = np.exp(-((period_obs[i] - Y)**2/2.0**2/((period_err[i])**2+V))) \
                / (period_err[i]+V)
            lik2 = np.sum(like2) / float(l2.sum())
        else:
            lik2 = 0.0

        # subgiants
        l3 = (logg_samp[i,:] < logg_cut)
        if l3.sum() > 0:
            like3 = np.exp(-((period_obs[i] - Z)**2/2.0**2/((period_err[i])**2+U))) \
                / (period_err[i]+U)
            lik3 = np.sum(like3) / float(l3.sum())
        else:
            lik3 = 0.0
        ll[i] = np.log10(lik1 + lik2 + lik3)
    return np.sum(ll)
