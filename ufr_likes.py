import numpy as np
import matplotlib.pyplot as pl

def log_period_model(par, log_age, temp, Tk):
    return par[0] + par[1] * log_age + par[2] * np.log10(Tk - temp)

def lnlike(par, log_age_samp, temp_samp, log_period_samp, logg_samp, \
               temp_obs, temp_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, Tk):
    nobs,nsamp = log_age_samp.shape
    log_period_pred = log_period_model(par[:4], log_age_samp, temp_samp, Tk)
    ll = np.zeros(nobs)
    Y, V = par[3], par[4]
    Z, U = par[5], par[6]
    X, W = par[7], par[8]
    ll = np.zeros(nobs)

#     logg_cut = 4.
    logg_cut = 3.7
#     period_cut = 5.
    period_cut = 2.

#     tplots1 = []
#     pplots1 = []
#     tplots2 = []
#     pplots2 = []
#     tplots3 = []
#     pplots3 = []
#     tplots4 = []
#     pplots4 = []
    for i in np.arange(nobs):

        # cool MS stars
#         turnoff = np.polyval(coeffs, temp_samp[i,:])-.1
        turnoff = np.polyval(coeffs, temp_samp[i,:])
#         l1 = (temp_samp[i,:] < Tk) * (logg_samp[i,:] > logg_cut) * ((10**log_period_samp[i,:])/(10**(log_age_obs[i,:])**.5189) > period_cut)
        l1 = (temp_samp[i,:] < Tk) * (logg_samp[i,:] > logg_cut) * ((10**log_period_samp[i,:])/(10**(log_age_samp[i,:])**.5189) > period_cut)
        if l1.sum() > 0:
            like1 = \
                np.exp(-((log_period_obs[i] - log_period_pred[i,l1])/2.0/log_period_err[i])**2) \
                / log_period_err[i]
            lik1 = np.sum(like1) / float(l1.sum())
        else:
            lik1 = 0.0

        # hot MS stars
#         l2 = (temp_samp[i,:] > Tk) * (logg_samp[i,:] > logg_cut) * ((10**log_period_samp[i,:])/(10**(log_age_obs[i,:])**.5189) > period_cut)
        l2 = (temp_samp[i,:] > Tk) * (logg_samp[i,:] > logg_cut) * ((10**log_period_samp[i,:])/(10**(log_age_samp[i,:])**.5189) > period_cut)
#         l2 = (temp_samp[i,:] > Tk)
        if l2.sum() > 0:
            like2 = np.exp(-((log_period_obs[i] - Y)**2/2.0**2/((log_period_err[i])**2+V))) \
                / (log_period_err[i]+V)
            lik2 = np.sum(like2) / float(l2.sum())
        else:
            lik2 = 0.0

        # subgiants
        l3 = (logg_samp[i,:] < logg_cut) * ((10**log_period_samp[i,:])/(10**(log_age_samp[i,:])**.5189) > period_cut)
        if l3.sum() > 0:
            like3 = np.exp(-((log_period_obs[i] - Z)**2/2.0**2/((log_period_err[i])**2+U))) \
                / (log_period_err[i]+U)
            lik3 = np.sum(like3) / float(l3.sum())
        else:
            lik3 = 0.0

        # UFRs
        l4 = ((10**log_period_samp[i,:])/(10**(log_age_samp[i,:])**.5189) < period_cut)
        if l4.sum() > 0:
            like4 = np.exp(-((log_period_obs[i] - Z)**2/2.0**2/((log_period_err[i])**2+U))) \
                / (log_period_err[i]+U)
            lik4 = np.sum(like4) / float(l4.sum())
        else:
            lik4 = 0.0

#         tplots1.append(temp_samp[i,:][l1])
#         pplots1.append(log_period_pred[i,:][l1])
#         tplots2.append(temp_samp[i,:][l2])
#         pplots2.append(log_period_pred[i,:][l2])
#         tplots3.append(temp_samp[i,:][l3])
#         pplots3.append(log_period_pred[i,:][l3])
#         tplots4.append(temp_samp[i,:][l4])
#         pplots4.append(log_period_pred[i,:][l4])

        ll[i] = np.log10(lik1 + lik2 + lik3 + lik4)

#     print 'plotting'
#     pl.clf()
#     ntplots1 = [i for s in tplots1 for i in s]
#     npplots1 = [i for s in pplots1 for i in s]
#     ntplots2 = [i for s in tplots2 for i in s]
#     npplots2 = [i for s in pplots2 for i in s]
#     ntplots3 = [i for s in tplots3 for i in s]
#     npplots3 = [i for s in pplots3 for i in s]
#     ntplots4 = [i for s in tplots4 for i in s]
#     npplots4 = [i for s in pplots4 for i in s]
#     pl.plot(ntplots1, 10**np.array(npplots1), 'k.')
#     pl.plot(ntplots2, 10**np.array(npplots2), 'r.')
#     pl.plot(ntplots3, 10**np.array(npplots3), 'g.')
#     pl.plot(ntplots4, 10**np.array(npplots4), 'y.')
#     pl.xlim(pl.gca().get_xlim()[::-1])
#     pl.savefig('check')
#     raw_input('enter')

    return np.sum(ll)

def neglnlike(par, log_age_samp, temp_samp, log_period_samp, logg_samp, \
        temp_obs, temp_err, log_period_obs, log_period_err, logg_obs, logg_err, coeffs, Tk):
    nobs,nsamp = log_age_samp.shape
    log_period_pred = log_period_model(par[:4], log_age_samp, temp_samp, Tk)
    ll = np.zeros(nobs)
    Y, V = par[3], par[4]
    Z, U = par[5], par[6]
    ll = np.zeros(nobs)

    for i in np.arange(nobs):

        # cool MS stars
        turnoff = np.polyval(coeffs, temp_samp[i,:])-.1
        l1 = (temp_samp[i,:] < Tk) * (logg_samp[i,:] > turnoff)
        if l1.sum() > 0:
            like1 = \
                np.exp(-((log_period_obs[i] - log_period_pred[i,l1])/2.0/log_period_err[i])**2) \
                / log_period_err[i]
            lik1 = np.sum(like1) / float(l1.sum())
        else:
            lik1 = 0.0

        # hot MS stars
        l2 = (temp_samp[i,:] > Tk) * (logg_samp[i,:] > turnoff)
        if l2.sum() > 0:
            like2 = np.exp(-((log_period_obs[i] - Y)**2/2.0**2/((log_period_err[i])**2+V))) \
                / (log_period_err[i]+V)
            lik2 = np.sum(like2) / float(l2.sum())
        else:
            lik2 = 0.0

        # subgiants
        # l3 = logg_samp[i,:] < 4.
        l3 = logg_samp[i,:] < turnoff
        if l3.sum() > 0:
            like3 = np.exp(-((log_period_obs[i] - Z)**2/2.0**2/((log_period_err[i])**2+U))) \
                / (log_period_err[i]+U)
            lik3 = np.sum(like3) / float(l3.sum())
        else:
            lik3 = 0.0

        ll[i] = np.log10(lik1 + lik2 + lik3)

    return -np.sum(ll)
