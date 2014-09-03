import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from teff_bv import teff2bv_orig, teff2bv_err
from sklearn.cross_validation import StratifiedKFold

# stratified k-fold is useful when you have populations with different numbers
# of members
def stratifiedkfold(folds):
    a, a_err, a_errp, a_errm, p, p_err, bv, bv_err, g, g_err, \
            g_errp, g_errm, flag = load_dat('ACNHPF', tn=False, cv=False)
    return StratifiedKFold(flag, folds)

def load_dat(fname, tn, cv):

#     KID[0], t[1], t_err[2], a[3], a_errp[4], a_errm[5], p[6], p_err[7], logg[8], logg_errp[9], logg_errm[10], feh[11], feh_err[12]
#     data = np.genfromtxt('/Users/angusr/Python/Gyro/data/all_astero.txt', skip_header=1).T
#     data = np.genfromtxt('/Users/angusr/Python/Gyro/data/garcia_all_astero.txt')
    data = np.genfromtxt('/Users/angusr/Python/Gyro/data/all_astero_plusgarcia.txt')
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
    flag = data[13][l]

    # convert temps to bvs
    bv_obs, bv_err = teff2bv_err(t, g, feh, t_err, .5*(g_errp+g_errm), feh_err) #FIXME: should really use asymmetric errorbars
    l = len(bv_obs)

    # add clusters FIXME: reddening
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
    flag = np.concatenate((flag, data[8]))

    # obviously comment these lines out if you want to use temps
    t = bv_obs
    t_err = bv_err

    # select star group
    flag -= 3
    flag[flag<0] = 0
    fnames = ['A', 'H', 'P', 'N', 'C', 'F']
    flist = []
    for i in range(len(fnames)):
        if fname.find(fnames[i]) >= 0:
            flist.append(i)
    l = (np.sum([flag == i for i in flist], axis=0)) == 1
    t = t[l]; t_err = t_err[l]
    p = p[l]; p_err = p_err[l]
    a = a[l]; a_errp = a_errp[l]; a_errm = a_errm[l]
    g = g[l]; g_errp = g_errp[l]; g_errm = g_errm[l]

    # reduce errorbars if they go below zero FIXME
    # This is only necessary if you don't use asym
    g_err = .5*(g_errp+g_errm)
    a_err = .5*(a_errp + a_errm)
    diff = a - a_err
    l = diff < 0
    a_err[l] = a_err[l] + diff[l] - np.finfo(float).eps

    if cv:
        print len(p[tn])
        return a[tn], a_err[tn], a_errp[tn], a_errm[tn], p[tn], p_err[tn], \
                t[tn], t_err[tn], g[tn], g_err[tn], g_errp[tn], g_errm[tn], flag[tn]
    print len(p)
    return a, a_err, a_errp, a_errm, p, p_err, t, t_err, g, g_err, g_errp, g_errm, flag

if __name__ == "__main__":

    a, a_err, a_errp, a_errm, p, p_err, bv, bv_err, g, g_err, g_errp, g_errm = load_dat('PF')

    pl.clf()
    pl.subplot(2, 1, 1)
    pl.errorbar(bv, p, xerr=bv_err, yerr=p_err, fmt='k.', capsize=0, ecolor='.7', \
            markersize=2)
    pl.xlabel('colour')
    pl.subplot(2, 1, 2)
    pl.errorbar(a, p, xerr=(a_errp, a_errm), yerr=p_err, fmt='k.', capsize=0, ecolor='.7', \
            markersize=2)
    pl.xlabel('age')
    pl.savefig('test')
