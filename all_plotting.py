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
#     data = np.genfromtxt('/Users/angusr/Python/Gyro/data/garcia_all_astero.txt')
    data = np.genfromtxt('/Users/angusr/Python/Gyro/data/all_astero_plusgarcia.txt')
    KID = data[0]
    t = data[1]
    p = data[6]
    g = data[8]
    print len(p)

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
    l = len(bv_obs)

    # remove clusters but leave field
    # add clusters FIXME: reddening
    data = np.genfromtxt("/Users/angusr/Python/Gyro/data/clusters.txt", skip_header=1).T
#     l = (data[4]!=1.1) * (data[4]!=0.588)
#     l = (data[4]!=1.1) * (data[4]!=0.588) * (data[4]!=0.5)
#     l1 = g < -10
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

#     # just_clusters_no_NGC
#     l = (a!=0.5) * (a!=0.588) * (a!=.625)
#     for b in range(-5,0):
#         l[b] = False
#     l = l==False

    # all_no_NGC
    l = a!=1.1

    t = t[l]; t_err = t_err[l]
    p = p[l]; p_err = p_err[l]
    a = a[l]; a_errp = a_errp[l]; a_errm = a_errm[l]
    g = g[l]; g_errp = g_errp[l]; g_errm = g_errm[l]

    # using clusters only
    # using field stars only
#     l = -5
#     t = t[l:]; t_err = t_err[l:]
#     p = p[l:]; p_err = p_err[l:]
#     a = a[l:]; a_errp = a_errp[l:]; a_errm = a_errm[l:]
#     g = g[l:]; g_errp = g_errp[l:]; g_errm = g_errm[l:]
#     # removing meibom
#     l = a!=1.1
#     t = t[l]; t_err = t_err[l]
#     p = p[l]; p_err = p_err[l]
#     a = a[l]; a_errp = a_errp[l]; a_errm = a_errm[l]
#     g = g[l]; g_errp = g_errp[l]; g_errm = g_errm[l]

    # reduce errorbars if they go below zero FIXME
    # This is only necessary if you don't use asym
    g_err = .5*(g_errp+g_errm)
    a_err = .5*(a_errp + a_errm)
    diff = a - a_err
    l = diff < 0
    a_err[l] = a_err[l] + diff[l] - np.finfo(float).eps

    print len(p)
    return a, a_err, a_errp, a_errm, p, p_err, t, t_err, g, g_err, g_errp, g_errm

if __name__ == "__main__":

    # test the hot stars
    a, a_err, a_errp, a_errm, p, p_err, bv, bv_err, g, g_err, g_errp, g_errm = load_dat()

    pl.clf()
    pl.subplot(2, 1, 1)
    pl.errorbar(bv, p, xerr=bv_err, yerr=p_err, fmt='k.', capsize=0, ecolor='.7', \
            markersize=2)
    pl.xlabel('colour')
    pl.subplot(2, 1, 2)
    pl.errorbar(a, p, xerr=(a_errp, a_errm), yerr=p_err, fmt='k.', capsize=0, ecolor='.7', \
            markersize=2)
    pl.xlabel('age')
    pl.show()
    pl.savefig('test')

#     c = .45; logg_cut = 4.
#     l = (bv < .45) * (g > logg_cut)


    # find the mean b-v uncertainty
    l = (bv_err!=0.01) * (bv_err!=0.001)
#      print np.mean(bv_err[l])

#     print g_err[l]
