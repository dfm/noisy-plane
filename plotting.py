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
    l = (xr > 1.)

#     # load list of MS stars
#     MSKID = np.genfromtxt("/Users/angusr/Python/Gyro/data/MS_stars.txt").T
#     print MSKID
#     for i in MSKID:
#         print KID[KID == i]
#     raw_input('enter')

    xr = data[1][l]
    yr = data[3][l]
    zr = data[13][l]*1000 # convert to myr
    xr_err = data[2][l]
    yr_err = data[4][l]
    zr_errp = data[14][l]*1000
    zr_errm = data[15][l]*1000

    # "for now replace values <= 0 with means"
    yr[yr <= 0.] = np.mean(yr[yr > 0.])
    zr[zr <= 0.] = np.mean(zr[zr > 0.])
    xr_err[xr_err <= 0.] = np.mean(xr_err[xr_err > 0.])
    yr_err[yr_err <= 0.] = np.mean(yr_err[yr_err > 0.])
    zr_errp[zr_errp <= 0.] = np.mean(zr_errp[zr_errp > 0.])
    zr_errm[zr_errm <= 0.] = np.mean(zr_errm[zr_errm > 0.])

    # "take logs"
    xr = np.log10(xr)
    zr = np.log10(zr) # convert to myr

    # logarithmic errorbars"
    xr_err = log_errorbar(xr, data[2][l], data[2][l])[0]
    zr_err, zr_errp, zr_errm  = log_errorbar(zr, zr_errp, zr_errm)

#     # make up observational uncertainties
#     N = len(xr)
#     xr_err = .1+.1*np.random.rand(N) # xr is log period
#     yr_err = .1+.1*np.random.rand(N) # yr is log teff
# #     zr_err = .1+.1*np.random.rand(N) #zr is log age
#     zr_err = 1.+1.*np.random.rand(N) #zr is log age

    return xr, yr, zr, xr_err, yr_err, zr_err

def log_errorbar(y, errp, errm):
    plus = y + errp
    minus = y - errm
#     minus[minus<0] = plus[minus<0]
    log_err = np.log10(plus/minus) / 2. # mean
    log_errp = np.log10(plus/y) # positive error
    log_errm = np.log10(y/minus) # negative error
    l = minus < 0 # Make sure ages don't go below zero!
    log_err[l] = log_errp[l]
    log_errm[l] = log_errp[l]
    return log_err, log_errp, log_errm

def plt(x, y, z, xerr, yerr, zerr, m, fname):

    xs = np.linspace(min(x), max(x), num=500)
    ys = np.linspace(min(y), max(y), num=500)
    zs = models.model(m, xs, ys)
    xr, yr, zr, xr_err, yr_err, zr_err = load()
    zp = models.model(m, xs, np.ones_like(xs)*6000.)
#     zp = zs
    zt = models.model(m, np.ones_like(ys)*np.log10(10), ys)
#     zt = zs

    pl.clf()
    pl.subplot(3,1,1)
    pl.errorbar(y, (10**z), xerr = yerr, yerr = 10**zerr, fmt = 'k.', capsize = 0, ecolor='0.5')
    pl.plot(ys, 10**zt, 'b-')
    pl.xlim(pl.gca().get_xlim()[::-1])
    pl.ylabel('age (Myr)')
    pl.xlabel('Teff')

    pl.subplot(3,1,2)
    pl.errorbar(10**z, (10**x), xerr = 10**zerr, yerr = 10**xerr, fmt = 'k.', capsize = 0, ecolor='0.5')
    pl.plot(10**zp, 10**xs, 'b-')
    pl.xlabel('age (Myr)')
    pl.ylabel('period')

    pl.subplot(3,1,3)
    pl.errorbar(y, (10**x), xerr = yerr, yerr = 10**xerr, fmt = 'k.', capsize = 0, ecolor='0.5')
    pl.plot(ys, 10**xs, 'b-')
    pl.xlim(pl.gca().get_xlim()[::-1])
    pl.xlabel('Teff')
    pl.ylabel('period')
    pl.subplots_adjust(hspace = 0.5)
    pl.savefig("%s"%fname)

#     pl.clf()
#     pl.errorbar(y, z, xerr=yerr, yerr=zerr, fmt='k.', capsize=0)
#     py = np.arange(5000.,6000.,10)
#     pl.plot(py, models.model(m, np.ones_like(py)*np.log10(5), py), "r-")
#     pl.plot(py, models.model(m, np.ones_like(py)*np.log10(10), py), "r-")
#     pl.plot(py, models.model(m, np.ones_like(py)*np.log10(20), py), "r-")
#     pl.savefig("a_vs_t")

# Generate some fake data set
def fake_data(m_true, N):

    rd = load()
    x = np.random.uniform(min(rd[0]), max(rd[0]), N) # log(period)
    y = np.random.uniform(min(rd[1]), max(rd[1]), N) # log(Teff)
#     z = models.g_model(m_true, x, y) # log(age)
    z = models.model(m_true, x, y) # log(age)

    # observational uncertainties.
    x_err = 0.1+0.1*np.random.rand(N)
    y_err = 0.1+0.1*np.random.rand(N)
    z_err = 0.1+0.1*np.random.rand(N)

    # add noise
    z_obs = z+z_err*np.random.randn(N)
    x_obs = x+x_err*np.random.randn(N)
    y_obs = y+y_err*np.random.randn(N)
    return x_obs, y_obs, z_obs, x_err, y_err, z_err

def plot3d(x1, y1, z1, x2, y2, z2, m, fig, colour, sv):
    fig = pl.figure(fig)
    ax = fig.gca(projection='3d')
    ax.scatter(x1, y1, z1, c = colour, marker = 'o')
    ax.scatter(x2, y2, z2, c = colour, marker = 'o')
    x_surf=np.r_[min(x1):max(x1):100j]
    y_surf=np.r_[min(y1):max(y1):100j]
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = models.model(m, x_surf, y_surf)
    ax.plot_surface(x_surf, y_surf, z_surf, alpha = 0.2)
    ax.set_xlabel('Rotational period (days)')
    ax.set_ylabel('B-V')
    ax.set_zlabel('Age (Gyr)')
    pl.show()
    pl.savefig(sv)
