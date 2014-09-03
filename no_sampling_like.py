import numpy as np
import matplotlib.pyplot as pl

def period_model(par, age, bv, c):
    return par[0] * (age*1e3)**par[1] * (bv-c)**par[2]

def nslnlike(par, age_samp, temp_samp, period_samp, logg_samp, age_obs, age_err, \
               temp_obs, temp_err, period_obs, period_err, logg_obs, logg_err, c):
    nobs, nsamp = age_samp.shape

    period_pred = period_model(par[:3], age_obs, temp_obs, c)
    Y, V = par[3], par[4]
    Z, W = par[5], par[6]
    X, U, P = par[7], par[8], par[9]
    logg_cut = 4.

    ll = np.zeros(nobs)
    for i in np.arange(nobs):

        # cool MS stars
        if (temp_obs[i] > c) and (logg_obs[i] > logg_cut):
            like11 = \
                np.exp(-.5*((period_obs[i] - period_pred[i])/period_err[i])**2) \
                / period_err[i]
            if np.isnan(like11):
                like11 = 0.
            like12 = \
                np.exp(-.5*((period_obs[i] - X)/(U + period_err[i]))**2) \
                / (U + period_err[i])
            if np.isnan(like12):
                like12 = 0.
            lik1 = (1-P)*like11 + P*like12
        else:
            lik1 = 0.0

        # hot MS stars
        if (temp_obs[i] < c) and (logg_obs[i] > logg_cut):
            lik2 = np.exp(-.5*((period_obs[i] - Y)/(period_err[i]+V))**2) \
                / (V + period_err[i])
            if np.isnan(lik2):
                lik2 = 0.
        else:
            lik2 = 0.0

        # subgiants
        if (logg_obs[i] < logg_cut):
            lik3 = np.exp(-.5*((period_obs[i] - Z)/(period_err[i]+W))**2) \
                / (W + period_err[i])
            if np.isnan(lik3):
                lik3 = 0.
        else:
            lik3 = 0.0

        ll[i] = np.log(lik1 + lik2 + lik3)

    loglike = np.sum(ll)
    if np.isnan(loglike) == True:
        loglike = -np.inf
    return loglike
