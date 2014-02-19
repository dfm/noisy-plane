import numpy as np
import matplotlib.pyplot as pl

def plt(x, y, z, xerr, yerr, zerr, m):
    a = y > 0.4
    b = y < 0.4

    m_true = [0.5189,  0.7725, 0.601, 0.4]
    xs = np.linspace(min(x), max(x), num=500)
    ys = np.linspace(min(y), max(y), num=500)
    zs = model(m_true, xs, ys)

    # load data
    data = np.genfromtxt('/users/angusr/python/gyro/data/data.txt').T
    cols = np.genfromtxt("/users/angusr/python/gyro/data/colours.txt")
    period = data[1]
    l = (period > 1.)
    period = np.log10(data[1][l])
    age = np.log10(data[13][l]*1000) # convert to myr
    bv = cols[1][l]

    pl.clf()
    pl.subplot(2,1,1)
    pl.errorbar(y[a], (10**z[a]), xerr = yerr[a], yerr = zerr[a], fmt = 'k.')
    pl.errorbar(y[b], (10**z[b]), xerr = yerr[b], yerr = zerr[b], fmt = 'r.')
    pl.plot(bv, 10**age, 'c.')
    pl.plot(ys, 10**zs, 'b-')
    pl.ylabel('age')
    pl.xlabel('colour')

    pl.subplot(2,1,2)
    pl.errorbar(10**z[a], (10**x[a]), xerr = zerr[a], yerr = xerr[a], fmt = 'k.')
    pl.errorbar(10**z[b], (10**x[b]), xerr = zerr[b], yerr = xerr[b], fmt = 'r.')
    pl.plot(10**age, 10**period, 'c.')
    pl.plot(10**zs, 10**xs, 'b-')
    pl.xlabel('age')
    pl.ylabel('period')
    pl.savefig("fakedata")
    return

def model(m, x, y):
#     return 1./m[0]*(x - np.log10(m[1]) - m[2]*np.log10(y - m[3]))
    return 1./m[0]*(x - np.log10(m[1]) - m[2]*np.log10(y))

# generative model
def g_model(m, x, y): # model computes log(t) from log(p) and bv
    z = np.ones_like(y)
    a = y > 0.4
    b = y < 0.4
    z[a] = 1./m[0] * (x[a] - np.log10(m[1]) - m[2]*np.log10(y[a]))
    z[b] = np.random.normal(3.5, 0.2, len(z[b]))
    return z

# Generate some fake data set
def fake_data(m_true, N):

    x = np.random.uniform(0.5, 1.8, N) # log(period)
    y = np.random.uniform(0.2, 1.,N) # colour
    z = g_model(m_true, x, y) # log(age)

    # observational uncertainties.
    x_err = 0.01+0.01*np.random.rand(N)
    y_err = 0.01+0.01*np.random.rand(N)
    z_err = 0.01+0.01*np.random.rand(N)

    # add noise
    z_obs = z+z_err*np.random.randn(N)
    x_obs = x+x_err*np.random.randn(N)
    y_obs = y+y_err*np.random.randn(N)
    return x, y, z, x_obs, y_obs, z_obs, x_err, y_err, z_err
