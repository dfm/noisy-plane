import numpy as np
import matplotlib.pyplot as pl
from colour_conversion import gr2bv, deredden
from teff_bv import teff2bv_err

# make up colour errors
c_err = .04

# make up period errors for M 34 - 5% (conservative), multiplicative
pe = .05

# make up logg and errors for cluster stars - want these all to be treated as MS
g = 4.5
g_err = .001

# add hyades
data = np.genfromtxt("/Users/angusr/Python/Gyro/data/hyades.txt", skip_header=2).T
bv = data[0]
bv_err = data[1]*4
p = data[2]
p_err = data[3]
a = data[4]
a_err = data[5]
a_errp = data[5]
a_errm = data[5]
feh = np.ones_like(data[0]*.15)
flag = np.ones_like(data[0])*4

pl.clf()
pl.errorbar(bv, p, xerr=bv_err, yerr=p_err, fmt='r.', ecolor='r')

# add praesepe
data = np.genfromtxt('/Users/angusr/Python/Gyro/data/praesepe.txt').T
EBV = 0.0133
B, V = data[5], data[6]
drbv = deredden(B, V, EBV)
# bv = np.concatenate((bv, (B-V)))
bv = np.concatenate((bv, drbv))
bv_err = np.concatenate((bv_err, np.ones_like(data[5])*c_err))
p = np.concatenate((p, 1./data[3]))
p_err = np.concatenate((p_err, (1./data[3])*(data[4]/data[3])))
# a = np.concatenate((a, np.ones_like(data[5])*.59))
a = np.concatenate((a, np.ones_like(data[5])*.588))
a_err = np.concatenate((a_err, np.ones_like(data[5])*.137))
a_errp = np.concatenate((a_errp, np.ones_like(data[5])*.137))
a_errm = np.concatenate((a_errm, np.ones_like(data[5])*.137))
feh = np.concatenate((feh, np.ones_like(data[5])*.15))
flag = np.concatenate((flag, np.ones_like(data[5])*5))

pl.clf()
pl.plot((B-V), 1./data[3], 'k.')
pl.plot(drbv, 1./data[3], 'r.')
# pl.show()

pl.errorbar((data[5]-data[6]), (1./data[3]), xerr=(np.ones_like(data[5])*c_err), yerr=((1./data[3])*(data[4]/data[3])), fmt='b.', ecolor='b')

# add NGC6811
data = np.genfromtxt("/Users/angusr/Python/Gyro/data/NGC6811.txt", skip_header=44).T
p = np.concatenate((p, data[3]))
p_err = np.concatenate((p_err, data[4]))
EBV = .1
mbv = gr2bv(data[1], data[2], EBV)
bv = np.concatenate((bv, mbv))
bv_err = np.concatenate((bv_err, np.ones_like(mbv)*c_err))
a = np.concatenate((a, np.ones_like(data[3])*1.1))
# a_err = np.concatenate((a_err, np.ones_like(data[3])*.1))
a_err = np.concatenate((a_err, np.ones_like(data[3])*.2))
feh = np.concatenate((feh, np.ones_like(data[1])*.15))
flag = np.concatenate((flag, np.ones_like(data[1])*6))

pl.errorbar(mbv, data[3], xerr=np.ones_like(mbv)*c_err, yerr=data[4], fmt='g.', ecolor='g')

# add Coma Berenices
data = np.genfromtxt("/Users/angusr/Python/Gyro/data/ComaBer_bv.txt").T
p = np.concatenate((p, data[0]))
p_err = np.concatenate((p_err, data[0]*pe))
bv = np.concatenate((bv, data[1]))
bv_err = np.concatenate((bv_err, np.ones_like(data[1])*c_err))
a = np.concatenate((a, np.ones_like(data[0])*.5))
a_err = np.concatenate((a_err, np.ones_like(data[0])*.1))
feh = np.concatenate((feh, np.ones_like(data[0])*.05))
flag = np.concatenate((flag, np.ones_like(data[0])*7))

pl.errorbar(data[1], data[0], xerr=np.ones_like(data[1])*c_err, yerr=data[0]*pe, fmt='y.', ecolor='y')

# # add the Pleiades
# data = np.genfromtxt("/Users/angusr/Python/Gyro/data/pleiades.txt", skip_header=1).T
# p = np.concatenate((p, data[1]))
# p_err = np.concatenate((p_err, data[2]))
# bv = np.concatenate((bv, data[3]-data[4]))
# bv_err = np.concatenate((bv_err, np.ones_like(data[1])*c_err))
# a = np.concatenate((a, np.ones_like(data[0])*.1))
# a_err = np.concatenate((a_err, np.ones_like(data[0])*.005))

# # M 34
# data = np.genfromtxt("/Users/angusr/Python/Gyro/data/M34.txt", skip_header=1).T
# p = np.concatenate((p, data[0]))
# p_err = np.concatenate((p_err, data[0]*pe))
# bv = np.concatenate((bv, data[1]))
# bv_err = np.concatenate((bv_err, np.ones_like(data[1])*c_err))
# a = np.concatenate((a, np.ones_like(data[0])*.225))
# a_err = np.concatenate((a_err, np.ones_like(data[0])*.025))

# add alpha cen ab
data = np.genfromtxt("/Users/angusr/Python/Gyro/data/alphacen.txt", skip_header=3).T
bv = np.concatenate((bv, data[0]))
bv_err = np.concatenate((bv_err, data[1]))
p = np.concatenate((p, data[2]))
p_err = np.concatenate((p_err, data[3]))
a = np.concatenate((a, data[4]))
a_err = np.concatenate((a_err, data[5]))
flag = np.concatenate((flag, np.ones_like(data[0])*8))

# add field stars (convert to lists first)
bv = list(bv)
bv_err = list(bv_err)
p = list(p)
p_err = list(p_err)
a = list(a)
a_err = list(a_err)
flag = list(flag)

# add 18sco
data = np.genfromtxt("/Users/angusr/Python/Gyro/data/18sco.txt", skip_header=2).T
bv.append(data[0])
bv_err.append(data[1])
p.append(data[2])
p_err.append(data[3])
a.append(data[4])
a_err.append(data[5])
flag.append(8)

# add 16cygB
data = np.genfromtxt("/Users/angusr/Python/Gyro/data/16CygB.txt", skip_header=2).T
bv.append(data[8])
bv_err.append(data[9])
p.append(data[2])
p_err.append(data[3])
a.append(data[4])
a_err.append(data[5])
flag.append(8)

bv = np.array(bv)
bv_err = np.array(bv_err)
p = np.array(p)
p_err = np.array(p_err)
a = np.array(a)
a_err = np.array(a_err)

# make up loggs
logg = np.ones_like(bv)*g
logg_err = np.ones_like(bv)*g_err

# logg[-5] = 4.57  # alpha cen b
# logg[-4] = 4.32  # alpha cen a
# logg[-3] = 4.41  # 18 Sco
# logg[-2] = 4.34  # 16 Cyg B
# logg[-1] = 4.44  # sun
#
# logg_err[-3] = 0.06  # 18 Sco
# logg_err[-2] = 0.07  # 16 Cyg B
# logg_err[-1] = 1e-5  # sun

data = np.zeros((9, len(bv)+1))
data[0,:-1] = bv
data[1,:-1] = bv_err
data[2,:-1] = p
data[3,:-1] = p_err
data[4,:-1] = a
data[5,:-1] = a_err
data[6,:-1] = logg
data[7,:-1] = logg_err
data[8,:-1] = flag

# add the sun
datasun = list(np.genfromtxt("/Users/angusr/Python/Gyro/data/sun.txt", skip_header=2).T)
datasun.append(4.5)
datasun.append(logg_err[-1])
datasun.append(8)
datasun = np.array(datasun)
data[:,-1] = datasun

np.savetxt("clusters.txt", np.transpose(data))
