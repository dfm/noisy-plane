import numpy as np
# import teff_bv

def model(m, x, y):
    Tk = 6500.
    return m[0]*x + m[1] + m[2]*np.log10(Tk-y)
#     return m[0]*x + m[1] + m[2]*np.log10(m[3]-y)

# generative model
def g_model(m, x, y): # model computes log(t) from log(p) and bv
    z = np.ones_like(y)
    cutoff = 6000.
    a = y > cutoff
    b = y < cutoff
    z[a] = model(m, x[a], y[a])
    z[b] = np.random.normal(m[4], m[5], len(z[b]))
    return z

# Barnes colour model
def bc_model(z, y):
    n, a, b, c = 0.5189, 0.7725, 0.601, 0.4
    return n*z + np.log10(a) + b*np.log10(y-c)

# Barnes teff model
def bt_model(z, y):
#     n, a, b, c = 0.5189, 0.7725, -0.00723, 6500.
    n, a, b, c = 0.5189, 0.7725, -0.06, 6500.
    # b = 0.601*np.log10(0.5-0.4)/np.log10(6500-6000)
    return n*z + np.log10(a) + b*np.log10(c-y)

# inverse Barnes colour model
def ibc_model(x, y):
    n, a, b, c = 0.5189, 0.7725, 0.601, 0.4
    return (1./n)*(x - np.log10(a) - b*np.log10(y-c))

# inverse Barnes teff model
def ibt_model(x, y):
    n, a, b, c = 0.5189, 0.7725, -0.06, 6500.
    return (1./n)*(x - np.log10(a) - b*np.log10(c-y))

# simpler inverse Barnes teff model
def sibt_model(x, y):
#     m = [1.927, 0.1121, -0.06, 6500.]
    m = [1.927, 0.216, 0.1156, 6500.]
#     return m[0]*(x + m[1] - m[2]*np.log10(m[3]-y))
    return m[0]*x + m[1] + m[2]*np.log10(m[3]-y)

# age = 4000
# teff = 5000
# bv = teff_bv.teff2bv(5000, 4.3, -0.2)
# b = 0.601*np.log10(bv-0.4)/np.log10(6500-6000) # = -0.00723
# period = 10
# print 'period = ', 10**(bc_model(np.log10(age), bv))
# print 'period = ', 10**(bt_model(np.log10(age), teff))
# print 'age = ', 10**(ibc_model(np.log10(period), bv))
# print 'age = ', 10**(ibt_model(np.log10(period), teff))
# print 'age = ', 10**(sibt_model(np.log10(period), teff))
#
