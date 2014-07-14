import numpy as np
import matplotlib.pyplot as pl

# everywhere i've written temp, i mean bv.

def period_model(par, age, bv, c):
    return par[0] * (age*1e3)**par[1] * (bv-c)**par[2]

def lnlike(par, age_samp, temp_samp, period_samp, logg_samp, age_obs, age_err, \
               temp_obs, temp_err, period_obs, period_err, logg_obs, logg_err, coeffs, c):
    nobs, nsamp = age_samp.shape

    period_pred = period_model(par[:3], age_samp, temp_samp, c)
    Y, V = par[3], par[4]
    Z, W = par[5], par[6]
    X, U, P = par[7], par[8], par[9]
    logg_cut = 4.

    ll = np.zeros(nobs)
    for i in np.arange(nobs):

        # cool MS stars
        l1 = (temp_samp[i,:] > c) * (logg_samp[i,:] > logg_cut)
        if l1.sum() > 0:
            like11 = \
                np.exp(-.5*((period_obs[i] - period_pred[i,l1])/period_err[i])**2) \
                / period_err[i]
            like11[np.isnan(like11)] = 0. # catching <0 samples
            like12 = \
                np.exp(-.5*((period_obs[i] - X)/(U + period_err[i]))**2) \
                / (U + period_err[i]) # FIXME here U is sig, not var
            like1 = (1-P)*like11 + P*like12
            loglike1 = np.log((1-P)*like11 + P*like12)
            loglik1 = np.logaddexp.reduce(loglike1, axis=0)/float(l1.sum())
            lik1 = np.sum(like1) / float(l1.sum())
        else:
            lik1 = 0.0

        # hot MS stars
        l2 = (temp_samp[i,:] < c) * (logg_samp[i,:] > logg_cut)
        if l2.sum() > 0:
            like2 = np.exp(-.5*((period_obs[i] - Y)/(period_err[i]+V)**2)) \
                / (V + period_err[i]) # FIXME now V is sig, not var
            loglike2 = np.log(like2)
            loglik2 = np.logaddexp.reduce(loglike2, axis=0) # don't divide by J!
            lik2 = np.sum(like2) / float(l2.sum())
        else:
            lik2 = 0.0

        # subgiants
        l3 = (logg_samp[i,:] < logg_cut)
        if l3.sum() > 0:
            like3 = np.exp(-.5*((period_obs[i] - Z)/(period_err[i]+U)**2)) \
                / (W + period_err[i]) # FIXME now W is sig, not var
            loglike3 = np.log(like3)
            loglik3 = np.logaddexp.reduce(loglike3, axis=0) # don't divide by J!
            lik3 = np.sum(like3) / float(l3.sum())
        else:
            lik3 = 0.0
        ll[i] = np.log10(lik1 + lik2 + lik3)
    return np.sum(ll)
