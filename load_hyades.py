import numpy as np
# from teff_bv import bv2teff

def hya_load():
    hya = np.genfromtxt("/Users/angusr/Python/Gyro/data/hyades.txt", skip_header=2).T
    sun = np.genfromtxt("/Users/angusr/Python/Gyro/data/sun.txt", skip_header=2).T
    sun[4] *= 1000.
    sun[5] *= 1000.

    data = np.empty((6, len(hya[0])+1))
    data[:4, 1:] = hya
    data[4, 1:] = np.ones_like(hya[0])*650
    data[5, 1:] = np.ones_like(hya[0])*data[4,1:]*.1
    data[:, :1] = sun[:, None]

#     data[0] = bv2teff(data[0])
#     data[1] = np.ones_like(data[0])*data[0]*.01

    # P, BV, A, Perr, BVerr, Aerr
    return data[2], data[0], data[4], data[3], data[1], data[5]

hya_load()
