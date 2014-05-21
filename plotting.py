import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from teff_bv import teff2bv

def load_dat():

    # "load data"
#     data = np.genfromtxt('/Users/angusr/Python/Gyro/data/data.txt').T
    data = np.genfromtxt('/Users/angusr/Python/Gyro/data/new_data.txt').T
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

    # convert temps to bvs
    bv_obs = teff2bv(t, g, np.ones_like(t)*-.2)

    # add praesepe
    data = np.genfromtxt("/Users/angusr/Python/Gyro/data/praesepe.txt").T
    period = 1./data[3]
    period_err = period*(data[4]/data[3])
    bv = data[5]-data[6]
    p = np.concatenate((p, period))
    p_err = np.concatenate((p_err, period_err))
    bv_obs = np.concatenate((bv_obs, bv))
    bv_err = np.ones_like(bv_obs)*0.01 # made up for now
    a = np.concatenate((a, np.ones_like(period)*.59))
    a_errp = np.concatenate((a_errp, np.ones_like(period)*.01)) # made up age errs
    a_errm = np.concatenate((a_errm, np.ones_like(period)*.01)) # made up age errs
    g = np.concatenate((g, np.ones_like(period)*4.5))
    g_errp = np.concatenate((g_errp, np.ones_like(p)*.001))
    g_errm = np.concatenate((g_errm, np.ones_like(p)*.001))

    # add hyades
    data = np.genfromtxt("/Users/angusr/Python/Gyro/data/hyades.txt", skip_header=2).T
    bv_obs = np.concatenate((bv_obs, data[0]))
    bv_err = np.concatenate((bv_err, data[1]))
    p = np.concatenate((p, data[2]))
    p_err = np.concatenate((p_err, data[3]))
    a = np.concatenate((a, data[4]))
    a_errp = np.concatenate((a_errp, data[5]))
    a_errm = np.concatenate((a_errm, data[5]))
    g = np.concatenate((g, np.ones_like(data[0])*4.5))
    g_errp = np.concatenate((g_errp, np.ones_like(data[0])*.001))
    g_errm = np.concatenate((g_errm, np.ones_like(data[0])*.001))

    # obviously comment these lines out if you want to use temps
    t = bv_obs
    t_err = bv_err

#     # 3d plot
#     p_cut = 5.
#     g_cut = 3.7
# #     t_cut = 6250
#     t_cut = .4
# #     coolMS = (t < t_cut) * (g > g_cut) * (p/a**.5189 > p_cut)
#     coolMS = (t > t_cut) * (g > g_cut) * (p/a**.5189 > p_cut)
# #     hotMS = (t > t_cut) * (g > g_cut) * (p/a**.5189 > p_cut)
#     hotMS = (t < t_cut) * (g > g_cut) * (p/a**.5189 > p_cut)
#     subs = (g < 4.) * (p/a**.5189 > p_cut)
#     ufrs = (p/a**.5189 < p_cut)
#     ufrsubs = (g > g_cut) * (p/a**.5189 > p_cut)
#     pl.clf()
#     fig = pl.figure()
#     ax = fig.gca(projection='3d')
#     ax.scatter(p[ufrs], t[ufrs], a[ufrs], c = 'b', marker = 'o')
#     ax.scatter(p[coolMS], t[coolMS], a[coolMS], c = 'r', marker = 'o')
#     ax.scatter(p[hotMS], t[hotMS], a[hotMS], c = 'g', marker = 'o')
#     ax.scatter(p[subs], t[subs], a[subs], c = 'y', marker = 'o')
#     ax.set_xlabel('Rotational period (days)')
#     ax.set_ylabel('Temp')
#     ax.set_zlabel('Age (Gyr)')
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


    # replace nans, zeros and infs in errorbars with means
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

    # take logs
    log_p = np.log10(p)
    log_a = np.log10(a)

    # logarithmic errorbars
    log_p_err = log_errorbar(p, p_err, p_err)[0]
    log_a_err, log_a_errp, log_a_errm  = log_errorbar(a, a_errp, a_errm)

    print log_p_err

#     log_p_err = np.log10(p_err)
#     log_a_err = np.log10(a_errp)

    # replace nans, zeros and infs in errorbars with means
    # find mean relative error
    log_a_err[np.isnan(log_a_err)] = np.mean(log_a_err[np.isfinite(log_a_err)])
    log_a_err[log_a_err==np.inf] = np.mean(log_a_err[np.isfinite(log_a_err)])
#     log_a_err[log_a_err<=0] = np.mean(log_a_err[log_a_err>0])
    log_p_err[np.isnan(log_p_err)] = np.mean(log_p_err[np.isfinite(log_p_err)])
#     log_p_err[log_p_err<=0] = np.mean(log_p_err[log_p_err>0])
    log_p_err[log_p_err==np.inf] = np.mean(log_p_err[np.isfinite(log_p_err)])

    # remove negative ages and infinite periods
#     a = (log_a > 0) * np.isfinite(log_p)

    a = np.isfinite(log_p)
    log_a = log_a[a]
    log_a_err = log_a_err[a]
    log_a_errp = log_a_errp[a]
    log_a_errm = log_a_errm[a]
    log_p = log_p[a]
    log_p_err = log_p_err[a]
    t = t[a]
    t_err = t_err[a]
    g = g[a]
    g_err = .5*(g_errp[a]+g_errm[a])
    g_errm = g_errm[a]
    g_errp = g_errp[a]

    # reduce errorbars if they go below zero
    diff = log_a - log_a_err
    a = diff < 0

    # really need to use asymmetric error bars!!!!
    log_a_err[a] = log_a_err[a] + diff[a] - np.finfo(float).eps #FIXME: might have to do this for neg_errs
#     a = log_p_err < 0.01 #FIXME: should be able to remove this line
#     log_p_err[a] = log_p_err[0]

    return log_p, t, log_a, log_p_err, t_err, log_a_err, log_a_errp, log_a_errm, g, g_err, g_errp, g_errm

def log_errorbar(y, errp, errm):
#     log_errp = (np.log10(y)*errp)/y
#     log_errm = (np.log10(y)*errm)/y
    for i in range(len(y)):
        print y[i], errp[i]
        print np.log10(y[i]), np.log10(errp[i])
        print np.log10(y[i]), np.log10((y[i]+errp[i])/y[i])
        raw_input('enter')
    log_errp = np.log10((y+errp)/y)
    log_errm = np.log10(y/(y-errm))
    log_err = .5*(log_errp + log_errm)
    return log_err, log_errp, log_err
