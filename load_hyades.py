import numpy as np

def hya_load():
    hya = np.genfromtxt("/Users/angusr/Python/Gyro/data/hyades.txt", skip_header=2).T
    sun = np.genfromtxt("/Users/angusr/Python/Gyro/data/sun.txt", skip_header=2).T

    data = np.empty((6, len(hya[0])+1))
    data[:4, 1:] = hya
    data[4, 1:] = np.ones_like(hya[0])*650
    data[5, 1:] = np.ones_like(hya[0])*10
    data[:, :1] = sun[:, None]

    return data[2], data[3], data[0], data[1], data[4], data[5]

hya_load()
