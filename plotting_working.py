import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from teff_bv import teff2bv

def load_dat():

    # "load data"
#     data = np.genfromtxt('/Users/angusr/Python/Gyro/data/data.txt').T
#     data = np.genfromtxt('/Users/angusr/Python/Gyro/data/new_data.txt').T
    data = np.genfromtxt('/Users/angusr/Python/Gyro/data/recovered.txt').T
#     data = np.genfromtxt('/Users/angusr/Python/Gyro/data/matched_data.txt').T
    KID = data[0]

    # check for duplicates
    n = 0
    for i in KID:
        match = np.where(KID == i)[0]
        if len(match) > 1:
            n+=1
    print n, 'duplicates found'

    p = data[1]
    t = data[3]
    g = data[10]

    # remove periods <= 0 and teff == 0
    l = (p > 0.)*(t > 0.)*(g > 0.)#*(g > 4.2)*(t < 6300)

    p = data[1][l]
    p_err = data[2][l]
    t = data[3][l]
    t_err = data[4][l]
    g = data[10][l]
    g_errp = data[11][l]
    g_errm = data[12][l]
    a = data[13][l]#*1000
    a_errp = data[14][l]#*1000
    a_errm = data[15][l]#*1000

    # 3d plot
    p_cut = 5.
    g_cut = 3.7
#     t_cut = 6250
    t_cut = .4
#     coolMS = (t < t_cut) * (g > g_cut) * (p/a**.5189 > p_cut)
    coolMS = (t > t_cut) * (g > g_cut) * (p/a**.5189 > p_cut)
#     hotMS = (t > t_cut) * (g > g_cut) * (p/a**.5189 > p_cut)
    hotMS = (t < t_cut) * (g > g_cut) * (p/a**.5189 > p_cut)
    subs = (g < 4.) * (p/a**.5189 > p_cut)
    ufrs = (p/a**.5189 < p_cut)
    ufrsubs = (g > g_cut) * (p/a**.5189 > p_cut)
    pl.clf()
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(p[ufrs], t[ufrs], a[ufrs], c = 'b', marker = 'o')
    ax.scatter(p[coolMS], t[coolMS], a[coolMS], c = 'r', marker = 'o')
    ax.scatter(p[hotMS], t[hotMS], a[hotMS], c = 'g', marker = 'o')
    ax.scatter(p[subs], t[subs], a[subs], c = 'y', marker = 'o')
    ax.set_xlabel('Rotational period (days)')
    ax.set_ylabel('Temp')
    ax.set_zlabel('Age (Gyr)')
#     pl.show()
#     raw_input('enter')

#     dnu = data[17][l]
#     dnu_err = data[18][l]

#     pl.clf()
#     pl.plot(t, dnu, 'k.')
#     pl.ylim(pl.gca().get_ylim()[::-1])
#     pl.xlim(pl.gca().get_xlim()[::-1])
#     pl.savefig('t_dnu')
#     raw_input('enter')

    # take logs
    log_p = np.log10(p)
    log_a = np.log10(a)

    # logarithmic errorbars
    log_p_err = log_errorbar(p, p_err, p_err)[0]
    log_a_err, log_a_errp, log_a_errm  = log_errorbar(a, a_errp, a_errm)

#     log_p_err = np.log10(p_err)
#     log_a_err = np.log10(a_errp)

    # replace nans, zeros and infs in errorbars with means
    # find mean relative error
    log_a_err[np.isnan(log_a_err)] = np.mean(log_a_err[np.isfinite(log_a_err)])
    log_a_err[log_a_err==np.inf] = np.mean(log_a_err[np.isfinite(log_a_err)])
    log_a_err[log_a_err<=0] = np.mean(log_a_err[log_a_err>0])
    log_p_err[np.isnan(log_p_err)] = np.mean(log_p_err[np.isfinite(log_p_err)])
    log_p_err[log_p_err<=0] = np.mean(log_p_err[log_p_err>0])
    log_p_err[log_p_err==np.inf] = np.mean(log_p_err[np.isfinite(log_p_err)])

    # remove negative ages and infinite periods
#     l = (log_a > 0) * np.isfinite(log_p)

    l = np.isfinite(log_p)
    log_a = log_a[l]
    log_a_err = log_a_err[l]
    log_a_errp = log_a_errp[l]
    log_a_errm = log_a_errm[l]
    log_p = log_p[l]
    log_p_err = log_p_err[l]
    t = t[l]
    t_err = t_err[l]
    g = g[l]
    g_err = .5*(g_errp[l]+g_errm[l])
#     g_err = .5*(g_errp+g_errm)
    g_errm = g_errm[l]
    g_errp = g_errp[l]

    # reduce errorbars if they go below zero
    diff = log_a - log_a_err
    l = diff < 0

    # really need to use asymmetric error bars!!!!
    log_a_err[l] = log_a_err[l] + diff[l] - np.finfo(float).eps #FIXME: might have to do this for neg_errs
#     l = log_p_err < 0.01 #FIXME: should be able to remove this line
#     log_p_err[l] = log_p_err[0]

    return log_p, t, log_a, log_p_err, t_err, log_a_err, log_a_errp, log_a_errm, g, g_err, g_errp, g_errm, a, a_errp, a_errm, p, p_err

def log_errorbar(y, errp, errm):
#     log_errp = (np.log10(y)*errp)/y
#     log_errm = (np.log10(y)*errm)/y
    log_errp = np.log10((y+errp)/y)
    log_errm = np.log10(y/(y-errm))
    log_err = .5*(log_errp + log_errm)
    return log_err, log_errp, log_err
