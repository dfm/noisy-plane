import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from teff_bv import teff2bv_orig, teff2bv_err

# def test_errorbar(x, x_errp, x_errm):
#     x = np.random.randn(1000)+100
#     pl.clf()
#     pl.hist(np.log(x), 50)
#     pl.savefig('test')
#     raw_input('enter')

def load_dat():

#     KID[0], t[1], t_err[2], a[3], a_errp[4], a_errm[5], p[6], p_err[7], logg[8], logg_errp[9], logg_errm[10], feh[11], feh_err[12]
#     data = np.genfromtxt('/Users/angusr/Python/Gyro/data/all_astero.txt', skip_header=1).T
    data = np.genfromtxt('/Users/angusr/Python/Gyro/data/garcia_all_astero.txt', skip_header=1).T
    KID = data[0]
    t = data[1]
    p = data[6]
    g = data[8]

    # remove periods <= 0 and teff == 0 FIXME: why is this necessary?
    l = (p > 0.)*(t > 0.)*(g > 0.)

    KID = data[0][l]
    p = p[l]
    p_err = data[7][l] # FIXME: check these!
    t = t[l]
    t_err = data[2][l]
    g = g[l]
    g_errp = data[9][l]
    g_errm = data[10][l]
    a = data[3][l]
    a_errp = data[4][l]
    a_errm = data[5][l]
    feh = data[11][l]
    feh_err = data[12][l]

    # convert temps to bvs
    bv_obs, bv_err = teff2bv_err(t, g, feh, t_err, .5*(g_errp+g_errm), feh_err) #FIXME: should really use asymmetric errorbars

    # add clusters FIXME: check all uncertainties on cluster values
    data = np.genfromtxt("/Users/angusr/Python/Gyro/data/clusters.txt", skip_header=1).T
    bv_obs = np.concatenate((bv_obs, data[0]))
    bv_err = np.concatenate((bv_err, data[1]))
    p = np.concatenate((p, data[2]))
    p_err = np.concatenate((p_err, data[3]))
    a = np.concatenate((a, data[4]))
    a_errp = np.concatenate((a_errp, data[5]))
    a_errm = np.concatenate((a_errm, data[5]))
    g = np.concatenate((g, data[6]))
    g_errp = np.concatenate((g_errp, data[7]))
    g_errm = np.concatenate((g_errm, data[7]))

    # obviously comment these lines out if you want to use temps
    t = bv_obs
    t_err = bv_err

    # reduce errorbars if they go below zero FIXME
    # This is only necessary if you don't use asym
    g_err = .5*(g_errp+g_errm)
    a_err = .5*(a_errp + a_errm)
    diff = a - a_err
    l = diff < 0
    a_err[l] = a_err[l] + diff[l] - np.finfo(float).eps

    return a, a_err, a_errp, a_errm, p, p_err, t, t_err, g, g_err, g_errp, g_errm
