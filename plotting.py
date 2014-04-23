import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
# from mixture import model
import models

def load_dat():

    # "load data"
    data = np.genfromtxt('/Users/angusr/Python/Gyro/data/data.txt').T
    KID = data[0]
    xr = data[1]

    # remove periods <= 0
    l = (xr > 0.)

    xr = data[1][l]
    yr = data[3][l]
    zr = data[13][l]
    xr_err = data[2][l]
    yr_err = data[4][l]
    zr_errp = data[14][l]
    zr_errm = data[15][l]

#     # "for now replace values <= 0 with means"
#     yr[yr <= 0.] = np.mean(yr[yr > 0.])
#     zr[zr <= 0.] = np.mean(zr[zr > 0.])
#     xr_err[xr_err <= 0.] = np.mean(xr_err[xr_err > 0.])
#     yr_err[yr_err <= 0.] = np.mean(yr_err[yr_err > 0.])
#     zr_errp[zr_errp <= 0.] = np.mean(zr_errp[zr_errp > 0.])
#     zr_errm[zr_errm <= 0.] = np.mean(zr_errm[zr_errm > 0.])

    # "take logs"
    xr = np.log10(xr)
    zr = np.log10(zr) # convert to myr

    # logarithmic errorbars"
    xr_err = log_errorbar(xr, data[2][l], data[2][l])[0]
    zr_err, zr_errp, zr_errm  = log_errorbar(zr, zr_errp, zr_errm)

    return xr, yr, zr, xr_err, yr_err, zr_err

def log_errorbar(y, errp, errm):
    log_errp = (np.log10(y)*errp)/y
    log_errm = (np.log10(y)*errm)/y
    log_err = .5*(log_errp + log_errm)
    return log_err, log_errp, log_err
