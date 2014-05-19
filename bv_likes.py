import numpy as np
import matplotlib.pyplot as pl

# everywhere I've written temp, i mean bv.

c = .4

def log_period_model(par, log_age, bv):
    return par[0] + par[1] * (log_age+3) + par[2] * np.log10(bv-c)

def lnlike(par, log_age_samp, temp_samp, log_period_samp, logg_samp, \
               temp_obs, temp_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, c):
    nobs,nsamp = log_age_samp.shape
    log_period_pred = log_period_model(par[:4], log_age_samp, temp_samp, c)
    ll = np.zeros(nobs)
    Y, V = par[3], par[4]
    Z, U = par[5], par[6]
    X, W, P = par[7], par[8], par[9]
    logg_cut = 3.7
#     logg_cut = 4.

    ll = np.zeros(nobs)
    for i in np.arange(nobs):

        # cool MS stars
#         turnoff = np.polyval(coeffs, temp_samp[i,:])
        l1 = (temp_samp[i,:] > c) * (logg_samp[i,:] > logg_cut)
        if l1.sum() > 0:
            like11 = \
                np.exp(-((log_period_obs[i] - log_period_pred[i,l1])/2.0/log_period_err[i])**2) \
                / log_period_err[i]
            like12 = \
                np.exp(-((log_period_obs[i] - X)**2/(2.0)**2/(W + log_period_err[i])**2)) \
                / (W + log_period_err[i]) # incorrect but works
#             like12 = \
#                 np.exp(-((log_period_obs[i] - X)**2/(2.0)**2/(log_period_err[i]**2+W))) \
#                 / (W + log_period_err[i]) # 'correct' but doesn't work
            like1 = (1-P)*like11 + P*like12 # FIXME: not sure if this line is correct
            lik1 = np.sum(like1) / float(l1.sum())
        else:
            lik1 = 0.0

        # hot MS stars
        l2 = (temp_samp[i,:] < c) * (logg_samp[i,:] > logg_cut)
        if l2.sum() > 0:
            like2 = np.exp(-((log_period_obs[i] - Y)**2/2.0**2/((log_period_err[i])**2+V))) \
                / (log_period_err[i]+V)
            lik2 = np.sum(like2) / float(l2.sum())
        else:
            lik2 = 0.0

        # subgiants
        l3 = (logg_samp[i,:] < logg_cut)
        if l3.sum() > 0:
            like3 = np.exp(-((log_period_obs[i] - Z)**2/2.0**2/((log_period_err[i])**2+U))) \
                / (log_period_err[i]+U)
            lik3 = np.sum(like3) / float(l3.sum())
        else:
            lik3 = 0.0

        ll[i] = np.log10(lik1 + lik2 + lik3)
    return np.sum(ll)
