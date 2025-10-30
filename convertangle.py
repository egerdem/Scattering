"""
Created on Tue Aug  7 23:37:30 2018

@author: Ege, Basic angle/coordinate conversions. 
"""
import pickle
import numpy as np

pi = np.pi


def degtorad(a):
    import numpy as np
    r = a * np.pi / 180
    return r


def radtodeg(a):
    import numpy as np
    r = a * 180 / np.pi
    return r


def sph2cart(th, ph, r=1.0):
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    ret = np.array([x, y, z])
    return (ret)


def cart2sph_single(xyz):
    """ converting a single row vector from cartesian to spherical coordinates """
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return (r, theta, phi)


def cart2sph(xyz):
    """ converting a multiple row vector matrix from cartesian to spherical coordinates """
    c = np.zeros((xyz.shape[0], 2))

    for i in range(xyz.shape[0]):
        x = xyz[i][0]
        y = xyz[i][1]
        z = xyz[i][2]

        r = np.sqrt(x * x + y * y + z * z)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        c[i][0] = theta
        c[i][1] = phi

    return [c]


def cart2sphr(xyz):
    """ converting a multiple row vector matrix
    from cartesian to spherical coordinates """
    c = np.zeros((xyz.shape[0], 2))
    r = []
    tt = []
    for i in range(xyz.shape[0]):
        x = xyz[i][0]
        y = xyz[i][1]
        z = xyz[i][2]

        rad = np.sqrt(x * x + y * y + z * z)
        tt.append(rad)
        theta = np.arccos(z / rad)
        phi = np.arctan2(y, x)
        c[i][0] = theta
        c[i][1] = phi
        r.append(rad)

    return [c, np.array(r)]

def cart2sphr_sparg(xyz):
    """ converting a multiple row vector matrix from cartesian to spherical coordinates """
    c = np.zeros((xyz.shape[0], 2))
    r = []
    tt = []
    for i in range(xyz.shape[0]):
        x = xyz[i][0]
        y = xyz[i][1]
        z = xyz[i][2]

        rad = np.sqrt(x * x + y * y + z * z)
        tt.append(rad)
        theta = np.arccos(z / rad)
        phi = np.arctan2(y, x)
        c[i][0] = theta
        c[i][1] = phi
        r.append(rad)
    return (np.array(r), c[:, 0], c[:, 1])

# r = 1
## upper loudspeakers
#
# r = 1.427 # origin to loudspeakers
# a = 1.5  #side length of the pentagon, meter
# l = 0.5*a/np.sin(np.radians(36))  #center to vertices length, pentagon
# dummy = np.sin(np.radians(54))*a/(2*np.sin(np.radians(36)))
# h = np.sqrt((a*np.sqrt(3)/2)**2-(a/(2*np.sin(np.radians(36)))-dummy)**2)
# t1 = np.arctan2(l,(h/2))
# t2 = np.pi - t1
# t3 = t2
# f1, f2, f3 = pi/2, 7*pi/10, 3*pi/10
# inc = 2*pi/5
# t1 = np.arctan2(l,(h/2))
# L1 = [r*np.sin(t1)*np.cos(f1), r*np.sin(t1)*np.sin(f1), r*np.cos(t1)]
# L7 = [r*np.sin(t1)*np.cos(f1+4*inc), r*np.sin(t1)*np.sin(f1+4*inc), r*np.cos(t1)]
# L8 = [r*np.sin(t1)*np.cos(f1+3*inc), r*np.sin(t1)*np.sin(f1+3*inc), r*np.cos(t1)]
# L9 = [r*np.sin(t1)*np.cos(f1+2*inc), r*np.sin(t1)*np.sin(f1+2*inc), r*np.cos(t1)]
# L10 = [r*np.sin(t1)*np.cos(f1+inc), r*np.sin(t1)*np.sin(f1+inc), r*np.cos(t1)]
#
## lower loudspeakers
# L6 = [r*np.sin(t2)*np.cos(f2+inc), r*np.sin(t2)*np.sin(f2+inc), r*np.cos(t2)]
# L5 = [r*np.sin(t2)*np.cos(f2+2*inc), r*np.sin(t2)*np.sin(f2+2*inc), r*np.cos(t2)]
# L4 = [r*np.sin(t2)*np.cos(f2+3*inc), r*np.sin(t2)*np.sin(f2+3*inc), r*np.cos(t2)]
# L2 = [r*np.sin(t2)*np.cos(f2), r*np.sin(t2)*np.sin(f2), r*np.cos(t2)]
# L3 = [r*np.sin(t3)*np.cos(f3), r*np.sin(t3)*np.sin(f3), r*np.cos(t3)]
#
# L = np.stack((L1,L2,L3,L4,L5,L6,L7,L8,L9,L10))
# dosya = open("L","wb")
# pickle.dump(L, dosya)
# dosya.close()
