import numpy as np

def model(m, x, y):
#     print "z", m[0]*x + m[1] + m[2]*np.log10(m[3]-y)
    return m[0]*x + m[1] + m[2]*np.log10(m[3]-y)

# generative model
def g_model(m, x, y): # model computes log(t) from log(p) and bv
    z = np.ones_like(y)
    cutoff = m[3]
    a = y > cutoff
    b = y < cutoff
    z[a] = model(m, x[a], y[a])
    z[b] = np.random.normal(m[4], m[5], len(z[b]))
    return z

def p_model(m, x):
    return m[0]*x

def t_model(m, y):
    return m[2]*np.log10(m[3]-y)
