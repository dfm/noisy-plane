import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from teff_bv import teff2bv_orig

def load_dat():

#     data = np.genfromtxt('/Users/angusr/Python/Gyro/data/new_matched.txt').T
    data = np.genfromtxt('/Users/angusr/Python/Gyro/data/p_errs.txt').T
    KID = data[0]

    # replace victor's stars
    vic = np.genfromtxt('/Users/angusr/Python/Gyro/data/Victor_params.txt', \
            skip_header=1).T

    for i in vic[0]:
        print KID[KID==i], i
        print data[1][KID==i], data[2][KID==i]
#         print data[1][KID==i], data[2][KID==i]
    raw_input('enter')

    # check for duplicates FIXME: sort this out!
    data2 = data
    print len(KID), 'targets found'
    matches = []
    for i, k in enumerate(KID):
        match = np.where(KID == k)[0]
        if len(match) > 1:
            matches.append(KID[match][0])
            data2[:,i] = np.zeros(len(data2[:,i]))
    print len(matches), 'duplicates found'

    # remove duplicates
    new_data = np.zeros((len(data2[:,0]), len(data2[0][data2[0] != 0])))
    new_data[i] = [data2[i][data2[0]!=0] for i in range(len(data2))]
#     data = new_data # comment this out if you don't want to remove duplicates
    print np.shape(data), 'targets remaining'

    p = data[1]
    t = data[3]
    g = data[10]

    # remove periods <= 0 and teff == 0 FIXME: why is this necessary?
    l = (p > 0.)*(t > 0.)*(g > 0.)#*(g > 4.2)*(t < 6300)

    p = data[1][l]
    p_err = data[2][l] # FIXME: check these!
    t = data[3][l]
    t_err = data[4][l]
    g = data[10][l]
    g_errp = data[11][l]
    g_errm = data[12][l]
    a = data[13][l]
    a_errp = data[14][l]
    a_errm = data[15][l]

    # convert temps to bvs
    bv_obs = teff2bv_orig(t, g, np.ones_like(t)*-.2)
    bv_err = np.ones_like(bv_obs)*0.01 # made up for now

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

    # replace nans, zeros and infs in errorbars with means FIXME: check why this is necessary
    # find mean relative error
    rel_a_errp = a[np.isfinite(a_errp)]/a_errp[np.isfinite(a_errp)]
    rel_a_errm = a[np.isfinite(a_errm)]/a_errm[np.isfinite(a_errm)]
    rel_p_err = p[np.isfinite(p_err)]/p_err[np.isfinite(p_err)]
    a_errp[np.isnan(a_errp)] = np.mean(rel_a_errp)*a[np.isnan(a_errp)]
    a_errm[np.isnan(a_errm)] = np.mean(rel_a_errm)*a[np.isnan(a_errm)]
    a_errp[a_errp==np.inf] = np.mean(rel_a_errp)*a[np.isinf(a_errp)]
    a_errm[a_errm==np.inf] = np.mean(rel_a_errm)*a[np.isinf(a_errm)]
    a_errp[a_errp<=0] = np.mean(rel_a_errp)*a[a_errp<=0]
    a_errm[a_errm<=0] = np.mean(rel_a_errm)*a[a_errm<=0]
    p_err[np.isnan(p_err)] = np.mean(rel_p_err)*p[np.isnan(p_err)]
    p_err[p_err<=0] = np.mean(rel_p_err)*p[p_err<=0]
    p_err[p_err==np.inf] = np.mean(rel_p_err)*p[p_err==np.inf]

    g_err = .5*(g_errp+g_errm)

    # reduce errorbars if they go below zero FIXME
    a_err = .5*(a_errp + a_errm)
    diff = a - a_err
    l = diff < 0
    a_err[l] = a_err[l] + diff[l] - np.finfo(float).eps

    # remove infinite periods FIXME: this shouldn't be necessary?
    l = np.isfinite(p)
    p = p[l]
    a = a[l]
    t = t[l]
    p_err = p_err[l]
    a_errp = a_errp[l]
    a_errm = a_errm[l]
    t_err = t_err[l]
    a_err = a_err[l]

    # try reducing the errors a little FIXME: this shouldn't be necessary
    p_err[p_err>10] = 5. #see if commenting this out helps?

    return a, a_err, a_errp, a_errm, p, p_err, t, t_err, g, g_err, g_errp, g_errm
