import numpy as np

def model(m, x, y):
#     print "y", y
#     print "z", m[0]*x + m[1] + m[2]*np.log10(m[3]-y)
    return m[0]*x + m[1] + m[2]*np.log10(m[3]-y)

# generative model
def g_model(m, x, y): # model computes log(t) from log(p) and bv
    z = np.ones_like(y)
    cutoff = 0.2
    a = y > cutoff
    b = y < cutoff
#     z[q] = m[0]*x[a] + m[1] + m[2]*(m[3]-y[a])
    z[a] = 1./m[0] * (x[a] - np.log10(m[1]) - m[2]*np.log10(m[3]-y[a]))
    mu = model(m, 1., cutoff)
    z[b] = np.random.normal(mu, 0.1, len(z[b]))
    return z

def p_model(m, x):
    return m[0]*x

def t_model(m, y):
    return m[2]*np.log10(m[3]-y)
