import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
# from mixture import model
import models

def load_dat():

    # "load data"
    data = np.genfromtxt('/Users/angusr/Python/Gyro/data/data.txt').T
    KID = data[0]
    p = data[1]
    t = data[3]
    g = data[10]

    # remove periods <= 0 and teff == 0
    l = (p > 0.)*(t > 0.)*(g > 4.1)*(t < 6400)

    p = data[1][l]
    p_err = data[2][l]
    t = data[3][l]
    t_err = data[4][l]
    g = data[10]
    g_errp = data[11]
    g_errm = data[11]
    a = data[13][l]#*1000
    a_errp = data[14][l]#*1000
    a_errm = data[15][l]#*1000

    # take logs
    log_p = np.log10(p)
    log_a = np.log10(a)

    # logarithmic errorbars
    log_p_err = log_errorbar(p, data[2][l], data[2][l])[0]
    log_a_err, log_a_errp, log_a_errm  = log_errorbar(a, a_errp, a_errm)

    # replace nans, zeros and infs in errorbars with means
    log_a_err[np.isnan(log_a_err)] = np.mean(log_a_err[np.isfinite(log_a_err)])
    log_a_err[log_a_err==np.inf] = np.mean(log_a_err[np.isfinite(log_a_err)])
    log_p_err[log_p_err<=0] = np.mean(log_p_err[log_p_err>0])

    # remove negative ages
    a = log_a > 0
    log_a = log_a[a]
    log_a_err = log_a_err[a]
    log_p = log_p[a]
    log_p_err = log_p_err[a]
    t = t[a]
    t_err = t_err[a]
    g = g[a]
    g_err = .5*(g_errp[a]+g_errm[a])

    # reduce errorbars if they go below zero
    diff = log_a - log_a_err
    a = diff < 0

    # really need to use asymmetric error bars!!!!
    log_a_err[a] = log_a_err[a] + diff[a] - np.finfo(float).eps
#     log_p_err = np.zeros_like(log_p) + 0.05
    a = log_p_err < 0.01
    log_p_err[a] = log_p_err[0]
#     log_p_err = np.ones_like(log_p_err)*log_p_err[0]
    print log_p_err
    raw_input('enter')

    return log_p, t, log_a, log_p_err, t_err, log_a_err, g, g_err

def log_errorbar(y, errp, errm):
    log_errp = (np.log10(y)*errp)/y
    log_errm = (np.log10(y)*errm)/y
    log_err = .5*(log_errp + log_errm)
    return log_err, log_errp, log_err
