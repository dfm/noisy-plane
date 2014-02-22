import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

def load():
    # load data
    data = np.genfromtxt('/users/angusr/python/gyro/data/data.txt').T
#     cols = np.genfromtxt("/users/angusr/python/gyro/data/colours.txt")
    xr = data[1]
    l = (xr > 1.)
    xr = np.log10(data[1][l])
    zr = np.log10(data[13][l]*1000) # convert to myr
#     yr = cols[1][l]
    yr = np.log10(data[3][l])

    # for now replace nans with means
    yr[np.isnan(yr)] = np.mean(yr[np.isfinite(yr)])
    zr[zr == -np.inf] = np.mean(zr[np.isfinite(zr)])

    # make up observational uncertainties
    N = len(xr)
    xr_err = 0.1+0.1*np.random.rand(N) # xr is log period
    yr_err = 0.1+0.1*np.random.rand(N) # yr is log teff
    zr_err = 0.1+0.1*np.random.rand(N) #zr is log age

    return xr, yr, zr, xr_err, yr_err, zr_err

def plt(x, y, z, xerr, yerr, zerr, m, fname):
#     a = y > 0.4
#     b = y < 0.4

    xs = np.linspace(min(x), max(x), num=500)
    ys = np.linspace(0.1, max(y), num=500)
    zs = model(m, xs, ys)

    xr, yr, zr, xr_err, yr_err, zr_err = load()

    pl.clf()
    pl.subplot(3,1,1)
#     pl.errorbar(y[a], (10**z[a]), xerr = yerr[a], yerr = zerr[a], fmt = 'k.')
#     pl.errorbar(y[b], (10**z[b]), xerr = yerr[b], yerr = zerr[b], fmt = 'r.')
#     pl.errorbar(y, (10**z), xerr = yerr, yerr = 10**zerr, fmt = 'k.', capsize = 0, ecolor='0.5') # use with colours
    pl.errorbar(10**y, (10**z), xerr = 10**yerr, yerr = 10**zerr, fmt = 'k.', capsize = 0, ecolor='0.5') # use with teff
#     pl.errorbar(y, z, xerr = yerr, yerr = zerr, fmt = 'k.', capsize = 0, ecolor='0.5')
#     pl.plot(yr, 10**zr, 'c.')
#     pl.plot(ys, 10**zs, 'b-')
    pl.plot(10**ys, 10**zs, 'b-')
#     pl.plot(ys, zs, 'b-')
    pl.ylabel('age')
    pl.xlabel('colour')

    pl.subplot(3,1,2)
#     pl.errorbar(10**z[a], (10**x[a]), xerr = zerr[a], yerr = xerr[a], fmt = 'k.')
#     pl.errorbar(10**z[b], (10**x[b]), xerr = zerr[b], yerr = xerr[b], fmt = 'r.')
    pl.errorbar(10**z, (10**x), xerr = 10**zerr, yerr = 10**xerr, fmt = 'k.', capsize = 0, ecolor='0.5')
#     pl.errorbar(z, x, xerr = zerr, yerr = xerr, fmt = 'k.', capsize = 0, ecolor='0.5')
#     pl.plot(10**zr, 10**xr, 'c.')
    pl.plot(10**zs, 10**xs, 'b-')
#     pl.plot(zs, xs, 'b-')
    pl.xlabel('age')
    pl.ylabel('period')

    pl.subplot(3,1,3)
#     pl.errorbar(y[a], (10**x[a]), xerr = zerr[a], yerr = xerr[a], fmt = 'k.')
#     pl.errorbar(y[b], (10**x[b]), xerr = zerr[b], yerr = xerr[b], fmt = 'r.')
#     pl.errorbar(y, (10**x), xerr = yerr, yerr = 10**xerr, fmt = 'k.', capsize = 0, ecolor='0.5')
    pl.errorbar(10**y, (10**x), xerr = 10**yerr, yerr = 10**xerr, fmt = 'k.', capsize = 0, ecolor='0.5')
#     pl.errorbar(y, x, xerr = yerr, yerr = xerr, fmt = 'k.', capsize = 0, ecolor='0.5')
#     pl.plot(yr, 10**xr, 'c.')
#     pl.plot(ys, 10**xs, 'b-')
    pl.plot(10**ys, 10**xs, 'b-')
#     pl.plot(ys, xs, 'b-')
    pl.xlabel('colour')
    pl.ylabel('period')
    pl.savefig("%s"%fname)

def model(m, x, y):
    return 1./m[0]*(x - np.log10(m[1]) - m[2]*np.log10(y))

def tmodel(m, x, y):
    return 1./m[0]*(x - np.log10(m[1]) - m[2]*y)

# generative model
def g_model(m, x, y): # model computes log(t) from log(p) and bv
    z = np.ones_like(y)
    cutoff = 0.2
    a = y > cutoff
    b = y < cutoff
    z[a] = 1./m[0] * (x[a] - np.log10(m[1]) - m[2]*np.log10(y[a]))
    mu = model(m, 1., cutoff)
    z[b] = np.random.normal(mu, 0.1, len(z[b]))
    return z

# Generate some fake data set
def fake_data(m_true, N):

    rd = load()

    x = np.random.uniform(min(rd[0]), max(rd[0]), N) # log(period)
    y = np.random.uniform(min(rd[1]), max(rd[1]), N) # colour
#     x = np.random.uniform(0.5, 1.8, N) # log(period)
#     y = np.random.uniform(0.2, 1.,N) # colour
    z = g_model(m_true, x, y) # log(age)

    # observational uncertainties.
#     x_err = 0.01+0.01*np.random.rand(N)
#     y_err = 0.01+0.01*np.random.rand(N)
#     z_err = 0.01+0.01*np.random.rand(N)
    x_err = 0.1+0.1*np.random.rand(N)
    y_err = 0.1+0.1*np.random.rand(N)
    z_err = 0.1+0.1*np.random.rand(N)

    # add noise
    z_obs = z+z_err*np.random.randn(N)
    x_obs = x+x_err*np.random.randn(N)
    y_obs = y+y_err*np.random.randn(N)
    return x, y, z, x_obs, y_obs, z_obs, x_err, y_err, z_err

def plot3d(x1, y1, z1, x2, y2, z2, m, fig, colour, sv):
    fig = pl.figure(fig)
    ax = fig.gca(projection='3d')
    ax.scatter(x1, y1, z1, c = colour, marker = 'o')
    ax.scatter(x2, y2, z2, c = colour, marker = 'o')
    x_surf=np.arange(min(x1), max(x1), 0.01)
    y_surf=np.arange(min(y1), max(y1), 0.01)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
#     m = [0.5189, 0.7725, 0.601]
#     m = [0.6, 0.5, 0.601]
    z_surf = model(m, x_surf, y_surf)
    ax.plot_surface(x_surf, y_surf, z_surf, alpha = 0.2)
    ax.set_xlabel('Rotational period (days)')
    ax.set_ylabel('B-V')
    ax.set_zlabel('Age (Gyr)')
    pl.show()
    pl.savefig(sv)
