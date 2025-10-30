# import reexpansioncoef as rxp
import Microphone as mc
import scipy.special as sp
import numpy as np
import recursion_r7 as recur
from numpy import matlib as mtlb
from scipy import linalg as LA
from pandas import *
import equal_area_sampling as sample
from pprint import pprint
# from InverseProblem import functions as ip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import figure, show, rc
import itertools


def planewave(freq, ra, A, thetas, phis, Nmax):
    """

    :param freq:
    :param ra:
    :param A:
    :param thetas:
    :param phis:
    :param Nmax:
    :return:
    """
    mvec = []
    kra = 2 * np.pi * freq / 340. * ra
    jn = sp.spherical_jn(Nmax, kra)
    jnp = sp.spherical_jn(Nmax, kra, derivative=True)
    yn = sp.spherical_yn(Nmax, kra)
    ynp = sp.spherical_yn(Nmax, kra, derivative=True)
    hn = jn - 1j * yn
    hnp = jnp - 1j * ynp
    bnkra = jn - (jnp / hnp) * hn
    b = []
    for n in range(Nmax+1):
        b.append(bnkra[n])

    for n in range(Nmax+1):
        for m in range(-n, n+1):
            pnm = A * 4 * np.pi * (1j)**n * b[n] * np.conj(sph_harm(m, n, phis, thetas))
            mvec.append(pnm)
    return np.array(mvec), freq


def cart2sph(x, y, z):
    """

    :param x: x-plane coordinate
    :param y: y-plane coordinate
    :param z: z-plane coordinate
    :return: Spherical coordinates (r, th, ph)
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    th = np.arccos(z / r)
    ph = np.arctan2(y, x)
    return r, th, ph


def sph2cart(r, th, ph):
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    u = np.array([x, y, z])
    return u

def shd_add_noise(n,m):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    wts = estr['weights']
    ths = estr['thetas']
    phs = estr['phis']
    Ynm = np.random.rand(32,)* 1j
    for ind in range(32):
        tq = ths[ind]
        pq = phs[ind]
        Ynm[ind] = sph_harm(m, n, pq, tq)
    return Ynm

def mic_loc(x,y,z):
    origin = [0,0,0]
    return np.array([x, y, z])


def mic_sub(mic_p, mic_q):
    mic_pq = mic_p - mic_q
    return mic_pq


def S_nm(n, m, k = 1, r = 1 , th = np.pi/2, ph = np.pi/2):
    return (sp.spherical_jn(n, k * r)+1j * sp.spherical_yn(n, k * r)) * sph_harm(m, n, ph, th)

def R_nm(n, m, k = 1, r = 1 , th = np.pi/2, ph = np.pi/2):
    return sp.spherical_jn(n, k * r) * sph_harm(m, n, ph, th)


def shd_nm(channels, n, m):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    wts = estr['weights']
    ths = estr['thetas']
    phs = estr['phis']
    pnm = np.zeros(np.shape(channels[0]))*1j
    for ind in range(32):
        cq = channels[ind]
        wq = wts[ind]
        tq = ths[ind]
        pq = phs[ind]
        Ynm = sph_harm(m, n, pq, tq) #Rafaely Ynm
        pnm += wq * cq * np.conj(Ynm)
    return pnm

def discorthonormality(N):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    wts = estr['weights']
    ths = estr['thetas']
    phs = estr['phis']
    CFnm = []
    for n in range(N+1):
        for m in range(-n, n+1):
            tmp = 0j
            for q in range(32):
                tmp += sph_harm(m, n, phs[q], ths[q]) * np.conj(sph_harm(m, n, phs[q], ths[q]))
            CFnm.append(1/tmp)
    return np.array(CFnm)

def shd_all(channels, Nmax, k, a):
    Pnm = []
    for n in range(Nmax+1):
        jn = sp.spherical_jn(n, k * a)
        jnp = sp.spherical_jn(n, k * a, derivative=True)
        yn = sp.spherical_yn(n, k * a)
        ynp = sp.spherical_yn(n, k * a, derivative=True)
        hn2 = jn + 1j * yn
        hn2p = jnp + 1j * ynp
        bnkra = jn - (jnp / hn2p) * hn2
        for m in range(-n, n+1):
            pnm = shd_nm(channels, n, m) * ((-1)**n) / (bnkra * 4 * np.pi * 1j**n)
            Pnm.append(pnm)
    return Pnm

def shd_all2(channels, Nmax, k, a):
    Pnm = []
    rho = 1.225
    c = 343
    for n in range(Nmax+1):
        jn = sp.spherical_jn(n, k * a)
        jnp = sp.spherical_jn(n, k * a, derivative=True)
        yn = sp.spherical_yn(n, k * a)
        ynp = sp.spherical_yn(n, k * a, derivative=True)
        hn2 = jn + 1j * yn
        hn2p = jnp + 1j * ynp
        bnkra = jn - (jnp / hn2p) * hn2
        for m in range(-n, n+1):
            pnm = shd_nm(channels, n, m) * jnp * k * a**2 / (-rho*c)
            Pnm.append(pnm)
    return np.array(Pnm) * discorthonormality(Nmax)

def Y_nm(n, m):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    wts = estr['weights']
    ths = estr['thetas']
    phs = estr['phis']
    Ynm = np.zeros(32)*1j
    for ind in range(32):
        tq = ths[ind]
        pq = phs[ind]
        Ynm[ind] = sph_harm(m, n, pq, tq)
    return Ynm


def sph_harm(m, n, phi, theta):
    sph = (-1)**m * np.sqrt((2*n+1)/(4*np.pi) * sp.factorial(n-np.abs(m)) / sp.factorial(n+np.abs(m))) * sp.lpmn(int(np.abs(m)), int(n), np.cos(theta))[0][-1][-1] * np.exp(1j*m*phi)
    return sph


def C_in_mono(N, freq, rs, rq_p):
    c = 343
    t_size = (N + 1) ** 2
    C_in = np.zeros(t_size) * 1j
    k = 2 * np.pi * freq / c
    rdiff = mic_sub(rs, rq_p)
    rdiff_sph = cart2sph(rdiff[0], rdiff[1], rdiff[2])
    for n in range(N):
        for m in range(-n, n+1):
            t = (n + 1) ** 2 - (n - m)
            Cnm = S_nm(n, -m, k, rdiff_sph[0], rdiff_sph[1], rdiff_sph[2]) * 1j * k
            C_in[t - 1] = Cnm
    return C_in


def C_in(N, rp_p, s, freq):
    rho = 1.225
    c = 343
    t_size = (N+1)**2
    C_input= np.zeros(t_size)*1j
    #print(C_in)
    k = 2 * np.pi * freq / c
    nvec = sph2cart(1, s[1], s[2])
    source_sph = cart2sph(-nvec[0],-nvec[1],-nvec[2])
    rp_p_cart = sph2cart(rp_p[0], rp_p[1], rp_p[2])
    k_vec = k*nvec
    phase = np.exp(np.sum(-k_vec*rp_p_cart)*1j)
    for n in range(N+1):
        #jn = sp.spherical_jn(n, k*r)
        #jn_p = sp.spherical_jn(n, k*r_p)
        for m in range(-n, n+1):
            Ynm_s = sph_harm(m, n, source_sph[2], source_sph[1])
            t = (n+1)**2 - (n-m)
            #print(t)
            anm = np.conj(Ynm_s)
            #anm = Ynm_s
            Cnm = anm * 4 * np.pi * (1j)**n * phase / (1j * 2 * np.pi * freq * -rho)  #* jn_p  * Ynm_qp/ (jn * Ynm_p-1j*w*rho) #phase eklenecek
            C_input[t-1] = Cnm
    return C_input



def L_dipole(n, a, k, rpq_p_sph):
    """
    Function to calculate coupled sphere (i.e. L12, L21)
    Utilisated to create L matrix elements
    :param n: Spherical harmonics order
    :param a: radius
    :param k: wave number
    :param rpq_p_sph: q pole to p pole distance (spherical coordinate)
    :return: L for two poles
    """
    n = n
    Lmax = n + 2
    Nmax = n + 2
    sqr_size = (Lmax - 1) ** 2
    jhlp = []
    L = np.zeros((n * (Lmax - 1) ** 2, n * (Lmax - 1) ** 2),
                 dtype=complex)  # np.full((n*(Lmax - 1) ** 2, n*(Lmax - 1) ** 2), np.arange(1.0,19.0))
    jhlp_fin = np.zeros((n * (Lmax - 1) ** 2, n * (Lmax - 1) ** 2), dtype=complex)  # L = np.arange(324.).reshape(18,18)

    l = n
    hnp_arr = []
    for i in range(l + 1):
        jnp = sp.spherical_jn(i, k * a, derivative=True)
        ynp = sp.spherical_yn(i, k * a, derivative=True)
        hnp = jnp + 1j * ynp
        for ind in range((i) * 2 + 1):
            jhlp.append(jnp / hnp)
            hnp_arr.append(hnp)
    jhlp = np.array(jhlp)
    # print("jhlp")
    # print(jhlp.shape)
    Ld = dict()
    Jhlpd = dict()
    L = np.eye(sqr_size)

    s = (n + 1) ** 2

    jhlp_fin = mtlb.repmat(jhlp, s, 1)
    hnp_fin = mtlb.repmat(hnp_arr, 1, 2)
    # print("jhlp_fin_shape")
    # print(jhlp_fin.shape)

    SR = recur.reexp_coef(k, Lmax, Nmax, rpq_p_sph)
    L = SR.copy() * jhlp_fin
    return L

def get_key(poles, val):
    for key, value in poles.items():
        if val[0] == value[0] and val[1] == value[1] and val[2] == value[2]:
            k = key
            return k

def L_multipole(ord, a, k, poles):
    """

    :param deg: Spherical harmonic order
    :param a: radius of sphere
    :param k: wave number
    :param poles: Coordinates of multipoles
    :return: Reexpension coefficient (SR) multipoles
    """
    sqr_size = (ord+1)**2 #(Lmax - 1) ** 2
    key, values = zip(*poles.items())
    Lsize = max(key)
    L_flat = np.array([])
    L_matrix = np.eye(sqr_size*Lsize, dtype=complex)
    for keys in itertools.product(values, repeat=2):
        if keys[0].all != keys[1].all:
            q = mic_loc(keys[0][0], keys[0][1], keys[0][2])
            p = mic_loc(keys[1][0], keys[1][1], keys[1][2])
            rq_p = mic_sub(q, np.array([0, 0, 0]))
            rp_p = mic_sub(p, np.array([0, 0, 0]))
            rpq_p = mic_sub(rq_p, rp_p)
            rqp_p = mic_sub(rp_p, rq_p)
            rpq_p_sph = cart2sph(rpq_p[0], rpq_p[1], rpq_p[2])
            L = L_dipole(ord, a, k, rpq_p_sph)
            L_flat = np.append(L_flat, L)
        else:
            L = np.eye(sqr_size)

            L_flat = np.append(L_flat, L)

    for row in key:
        for col in key:
            if row == col:
                L = np.eye(sqr_size)
            else:
                mq = poles.get(row)
                mp = poles.get(col)

                q = mic_loc(mq[0], mq[1], mq[2])
                p = mic_loc(mp[0], mp[1], mp[2])
                rq_p = mic_sub(q, np.array([0, 0, 0]))
                rp_p = mic_sub(p, np.array([0, 0, 0]))

                rpq_p = mic_sub(rq_p, rp_p)

                rpq_p_sph = cart2sph(rpq_p[0], rpq_p[1], rpq_p[2])
                L = L_dipole(ord, a, k, rpq_p_sph)

            L_matrix[((row-1) * sqr_size):((row) * sqr_size), ((col-1) * sqr_size):((col) * sqr_size)] = L
    print("COND")
    print(np.linalg.cond(L_matrix))
    return L_matrix


def D_multipole(n, a, freq, s_sph, k, poles):
    """

    :param n: Spherical harmonics order
    :param a: Radius of sphere
    :param freq: frequency
    :param s_sph: source spherical coordinates
    :param k: wave number
    :param poles: multipoles' spherical coordinates
    :return: D vector
    """
    m = (n+1)**2
    key, values = zip(*poles.items())
    Lsize = max(key)
    D_flat = np.array([])

    jhnp = []
    for i in range(n + 1):
        jnp = sp.spherical_jn(i, k * a, derivative=True)
        ynp = sp.spherical_yn(i, k * a, derivative=True)
        hnp = jnp + 1j * ynp
        for i in range((i) * 2 + 1):
            jhnp.append(jnp / hnp)

    jhnp = np.array(jhnp)
    for keys in itertools.product(values):
        q = mic_loc(keys[0][0], keys[0][1], keys[0][2])

        rq_p = mic_sub(q, np.array([0, 0, 0]))
        rq_p_sph = cart2sph(rq_p[0], rq_p[1], rq_p[2])

        C = C_in(n, rq_p_sph, s_sph, freq)

        Dp = C * -jhnp
        D_flat = np.append(D_flat, Dp)
    D_matrix = (D_flat.reshape(Lsize, m))

    return D_flat, jhnp

def A_multipole(L, D, n):
    """

    :param L: Reexpension coefficient matrix
    :param D: 
    :param n:
    :return:
    """
    lu, piv = LA.lu_factor(L)
    A_nm = LA.lu_solve((lu, piv), D) #Eq.39
    size = (n + 1) ** 2

    return A_nm[0:size], A_nm #ok


def pressure_withA(n_low, a, k, Anm):
    """
    Calculates pressure from spherical harmonic coefficients
    :param n_low: Spherical harmonics order
    :param a: Radius of sphere
    :param k: wave number
    :param Anm: Spherical Harmonics Coefficient (Matrix form)
    :return: Pressure for each microphones
    """
    rho = 1.225     # Density of air
    c = 343         # Speed of sound
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    mic32 = np.zeros(32)*1j
    for ind in range(32):
        tq = ths[ind]
        pq = phs[ind]
        potential = 0
        for n in range(0, n_low+1):
            jnp = sp.spherical_jn(n, k * a, derivative=True)
            for m in range(-n,n+1):
                Ynm = sph_harm(m, n, pq, tq)
                t = (n+1)**2 - (n-m) - 1
                potential += Anm[t]*Ynm / (jnp) #sph_harm(m, n, pw, tw) #Gumerov, Eq. 18
        pressure = -potential * c * rho / (k * a**2)
        mic32[ind] = pressure
    return mic32

def pressure_withA_multipole(n_low, a, k, Anm_all, no_of_poles):
    """
    Calculates pressure from spherical harmonic coefficients
    :param n_low: Spherical harmonics order
    :param a: Radius of sphere
    :param k: wave number
    :param Anm: Spherical Harmonics Coefficient (Matrix form)
    :return: Pressure for each microphones
    """
    rho = 1.225     # Density of air
    c = 343         # Speed of sound
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    pressure_all = []
    size = (n_low + 1) ** 2
    for arr in range(no_of_poles):
        Anm = Anm_all[arr*size:arr*size+size]
        #print(Anm)
        mic32 = np.zeros(32) * 1j
        for ind in range(32):
            tq = ths[ind]
            pq = phs[ind]
            potential = 0
            for n in range(0, n_low+1):
                jn = sp.spherical_jn(n, k * a)
                jnp = sp.spherical_jn(n, k * a, derivative=True)
                yn = sp.spherical_yn(n, k * a)
                ynp = sp.spherical_yn(n, k * a, derivative=True)
                hn2 = jn + 1j * yn
                hn2p = jnp + 1j * ynp
                bnkra = jn - (jnp / hn2p) * hn2

                jnp = sp.spherical_jn(n, k * a, derivative=True)
                for m in range(-n,n+1):
                    Ynm = sph_harm(m, n, pq, tq)
                    t = (n+1)**2 - (n-m) - 1
                    #print(np.abs(1/jnp))
                    potential += Anm[t] * Ynm / (jnp)  #sph_harm(m, n, pw, tw) #Gumerov, Eq. 30
            pressure = -potential * c * rho / (k * (a**2))
            #pressure = potential / (1j * k**2  * (a**2))
            # pressure = potential
            mic32[ind] = pressure
        #print(mic32)
        pressure_all.append(mic32)
    #print(pressure_all)
    return pressure_all

def add_noise(pressure, SNR, no_of_poles):
    """

    :param mic32: pressure at each mic
    :param SNR: Signal-to-Noise ratio (dB)
    :return:
    """
    for i in range(no_of_poles):
    #Ynm = np.random.rand(32, )
        pres_temp = pressure[i]
        #print("PRESSURE")
        #print(pres_temp)
        noise = np.random.rand(32,)
        noise_r = np.random.randn(32)
        noise_i = np.random.randn(32)
        noise_r = noise_r - np.mean(noise_r)
        noise_i = noise_i - np.mean(noise_i)
        noise = noise_r + noise_i * 1j
        #print("NOISE")
        #print(noise)

        #print("NORM")
        mic_norm = np.linalg.norm(pres_temp, axis= 0)
        #print(np.linalg.norm(pres_temp, axis= 0))
        #print(pres_temp)
        #print(len(pres_temp))
        noisy_pres = 0
        #print("NOISE NORM")
        noise_norm = np.linalg.norm(noise, axis=0)
        #print(noise_norm)
        coef = mic_norm/noise_norm
        SNR_linear = 10**(-SNR/20)
        #print("SNR")
        #print(SNR_linear)
        noise = noise * coef * SNR_linear
        #print(noise)
        #print(np.linalg.norm(noise, axis=0))
        pressure[i] = pres_temp + noise
    noisy_pres = pressure
    return noisy_pres

def pressure_to_Anm(pressure, n_max, no_of_poles):
    rho = 1.225
    Anm_scatter = []
    size = 32
    for arr in range(no_of_poles):
        pressure_temp = pressure[arr]
        Anm_scat_temp = np.array(shd_all2(pressure_temp, n_max, k, a)).flatten()
        Anm_scatter.append(Anm_scat_temp)
    Anm_scatter = np.array(Anm_scatter).flatten()
    return Anm_scatter


def Anm_to_D(Anm, L):
    D = np.dot(L, Anm)
    return D


def D_to_Cin(D, jhnp ):
    C_in_scat = D * (1 / -jhnp)
    # C_in_scat = D * (1 / -1)

    return C_in_scat


def idealYnm_conj(n_max, f, th_s, phi_s, rp_p):
    c = 343
    k = 2*np.pi*f/c
    nvec = sph2cart(1, th_s, phi_s)
    k_vec = k*nvec
    phase = np.exp(np.sum(-k_vec*rp_p)*1j)
    Ynm = []
    for n in range(0, n_max+1):
        for m in range(-n, n+1):
            Ynm.append(np.conj(sph_harm(m, n, phi_s, th_s))*phase)
    return np.array(Ynm)



def Ynm_err(n_max, f, th_s, phi_s, rp_p, pres, k, a):
    """
    Error Calculations
    :param n_max:
    :param f:
    :param th_s:
    :param phi_s:
    :param rp_p:
    :param pres:
    :param k:
    :param a:
    :return:
    """
    rho = 1.225
    shd_ideal = idealYnm_conj(n_max, f, th_s, phi_s, rp_p)
    #print("pressure")
    #print(pres)
    shd_scat = np.array(shd_all(pres, n_max, k, a)).flatten()/(2.56*-1j*k*343*rho) #no bnkr no phase


    #print("ideal")
    #print(shd_ideal)
    #print("scatterYnm")
    #print(shd_scat)
    #print("Ynmratio")
    #ratio = abs(shd_ideal[0]/shd_scat[0])
    #print(abs(ratio))
    s = 4
    err = np.linalg.norm(shd_ideal[s] - shd_scat[s]) / np.linalg.norm(shd_ideal[s])
    #print(10*np.log(err**2))
    error = 10*np.log10(err**2)

    #print("Ynmdiff")
    #diff = shd_ideal*2.56 - (shd_scat / (-1j * k * 343 * rho))
    #print(diff)
    #print(ratio/ratio[0])
    return error

def C_in_error(C_in, C_tilde):
    err = np.linalg.norm(C_in[0:9] - C_tilde[0:9]) / np.linalg.norm(C_in[0:9])
    #print(10*np.log(err**2))
    error = 10*np.log10(err**2)
    return error



if __name__=='__main__':
    n = 2  #Spherical Harmonic Order
    a = 42e-3   #Radius of sphere
    f = 5000    #Freq
    c = 343     #Speed of sound
    k = 2*np.pi*f/c #wave number
    ord = 2   # Spherical Harmonic Order for L matrix

    source = mic_loc(3, 3, 0 )

    rs = mic_sub(source, np.array([0, 0, 0]))

    ### ADD NOISE ###


    # mics = {1: mic_loc(0, 0.5, 0), 2: mic_loc(0, -0.5, 0), 3: mic_loc(0, 1.5, 0)}#, 4: mic_loc(0.0, -0.25, 0)} #, 5: mic_loc(0, 0.25, 0)}
    # mics = {1: mic_loc(0.5, 0, 0), 2: mic_loc(1.5, 0.0, 0), 3: mic_loc(1.0, 0.866, 0), 4: mic_loc(-0.5, 0, 0), 5: mic_loc(0, 0.866, 0), 6: mic_loc(0, -0.866, 0), 7: mic_loc(1, -0.866, 0)}

    ### SCENE SETUP ###
    center_x, center_y = 1., 1.

    """
    ### TRIANGLE ###
    locs = sample.tri_pole(1, 1, 0)
    mics = {1: mic_loc(center_x, center_y, 0),
            2: mic_loc(locs[0][0], locs[0][1], locs[0][2]),
            3: mic_loc(locs[1][0], locs[1][1], locs[1][2]),
            4: mic_loc(locs[2][0], locs[2][1], locs[2][2])}
    # mics = {1: mic_loc(0.5, 0, 0), 2: mic_loc(1.5, 0.0, 0), 3: mic_loc(-0.5, 0, 0)}
    

    ### SQUARE ###
    locs = sample.square_pole(1,1,0)
    mics = {1: mic_loc(center_x, center_y, 0),
            2: mic_loc(locs[0][0], locs[0][1], locs[0][2]),
            3: mic_loc(locs[1][0], locs[1][1], locs[1][2]),
            4: mic_loc(locs[2][0], locs[2][1], locs[2][2]),
            5: mic_loc(locs[3][0], locs[3][1], locs[3][2])}


    
    ### PENTA ###
    locs = sample.penta_pole(1,1,0)
    mics = {1: mic_loc(center_x, center_y, 0),
            2: mic_loc(locs[0][0], locs[0][1], locs[0][2]),
            3: mic_loc(locs[1][0], locs[1][1], locs[1][2]),
            4: mic_loc(locs[2][0], locs[2][1], locs[2][2]),
            5: mic_loc(locs[3][0], locs[3][1], locs[3][2]),
            6: mic_loc(locs[4][0], locs[4][1], locs[4][2])}
    """
    """
    ### HEXA ###
    locs = sample.hexa_pole(1,1,0)
    mics = {1: mic_loc(center_x, center_y, 0),
            2: mic_loc(locs[0][0], locs[0][1], locs[0][2]),
            3: mic_loc(locs[1][0], locs[1][1], locs[1][2]),
            4: mic_loc(locs[2][0], locs[2][1], locs[2][2]),
            5: mic_loc(locs[3][0], locs[3][1], locs[3][2]),
            6: mic_loc(locs[4][0], locs[4][1], locs[4][2]),
            7: mic_loc(locs[5][0], locs[5][1], locs[5][2])}
    """


    mics = {1: mic_loc(0.5, 0, 1), 2: mic_loc(1.5, 0.0, -1), 3: mic_loc(-0.5, 0, 3)}
    mics = {1: mic_loc(0.5, 0, 1), 2: mic_loc(1.5, 0.0, -1)}

    mic_test = mic_sub(mics.get(1), np.array([0, 0, 0]))
    print("mic")
    print(mic_test)

    key, values = zip(*mics.items())
    no_of_poles = max(key)

    
    #src = mic_loc(3, 3, 2)
    rsrc = mic_sub(source, np.array([0, 0, 0]))
    rsrc_sph = cart2sph(rs[0], rs[1], rs[2])
    L = L_multipole(ord, a, k, mics)
    D, jhnp = D_multipole(n, a, f, rsrc_sph, k, mics)
    A, Anm_all = A_multipole(L, D, n)
    pres_single = pressure_withA(n, a, k, A)
    #print("PRESSINGLE")
    #print(pres_single)
    presmulti = pressure_withA_multipole(n, a, k, Anm_all, no_of_poles)
    # NOISE ADDITION
    # noise_presmulti = add_noise(presmulti, 6, no_of_poles)
    # Anm_tilde = pressure_to_Anm(noise_presmulti, n, no_of_poles)

    Anm_tilde = pressure_to_Anm(presmulti, n, no_of_poles)

    # C_tilde
    size = (n+1)**2
    #Anm_all[0:size] = Anm_tilde
    jhnp = np.resize(jhnp, len(Anm_tilde))
    D_tilde = Anm_to_D(Anm_tilde, L)
    C_in_tilde = D_to_Cin(D_tilde, jhnp)
    C_in = D_to_Cin(D, jhnp)
    k = 0
    #print("Break")
    #print(A)
    #print("Break")
    #print(Anm_tilde)
    print("C-input")
    print(C_in)
    print("C-output")
    print(C_in_tilde)
    print("Diff")
    C_diff = C_in - C_in_tilde
    res = L[k]
    
    print("magnitude Diff")
    print(DataFrame(np.linalg.norm(C_diff[i]) for i in range(len(C_diff))))

    # For printing array as DataFrame
    desired_width = 600

    pandas.set_option('display.width', desired_width)

    np.set_printoptions(linewidth=desired_width)

    pandas.set_option('display.max_columns', 20)

    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    #.set_option('display.max_columns', 20)
    print(DataFrame(C_diff))
    #print(DataFrame(C_in_tilde))
    #CFnm = discorthonormality(4)
    #print(CFnm)

    """
    # ERROR PLOTTING C_IN & C_OUT
    poles = []
    angles = np.arange(0.0, 2 * np.pi, 2 * np.pi / 360)
    for angle in angles:
        # print(angle)
        # poles.append(mic_loc(np.cos(angle) * 5, np.sin(angle) * 5, 0))
        rs = mic_sub(mic_loc(np.cos(angle) * 3, np.sin(angle) * 3, 0), np.array([0, 0, 0]))
        rs_sph = cart2sph(rs[0], rs[1], rs[2])
        # print(rs_sph)
        poles.append(rs_sph)
    sources = np.array(poles)

    error_all = []
    for rss in sources:
        L = L_multipole(ord, a, k, mics)
        D, jhnp = D_multipole(n, a, f, rss, k, mics)
        A, Anm_all = A_multipole(L, D, n)
        presmulti = pressure_withA_multipole(n, a, k, Anm_all, no_of_poles)
        # NOISE ADDITION
        noise_presmulti = add_noise(presmulti, 6, no_of_poles)
        Anm_tilde = pressure_to_Anm(noise_presmulti, n, no_of_poles)

        #Anm_tilde = pressure_to_Anm(presmulti, n, no_of_poles)

        # C_in & C_tilde Calc
        size = (n + 1) ** 2
        # Anm_all[0:size] = Anm_tilde
        jhnp = np.resize(jhnp, len(Anm_tilde))
        D_tilde = Anm_to_D(Anm_tilde, L)
        C_in_tilde = D_to_Cin(D_tilde, jhnp)
        C_input = D_to_Cin(D, jhnp)

        # print(A)
        #presmulti = pressure_withA(n, a, k, A)
        # print("PRES:")
        # print(presmulti)
        # pres = pressure(n, a, f, rss, k, rp_p_sph, rq_p_sph, rpq_p_sph, rqp_p_sph)
        #err = Ynm_e rr(n, f, rss[1], rss[2], mic_test, presmulti, k, a)
        err = C_in_error(C_input, C_in_tilde)
        print(err)
        error_all.append(err)

    min = np.amin(error_all)
    max = np.amax(error_all)
    error_avg = sum(error_all)/360
    print("ERROR AVERAGE")
    print(error_avg)
    # PLOT
    fig = figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    # ax.set_rorigin(min)
    inc = abs(min - max)
    print("inc")
    print(inc)
    N = 360
    ax.set_ylim(min, max + inc / 8)
    theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
    radii = np.array(error_all)  # +abs(min)  # 10 * np.random.rand(N)
    width = 2 * np.pi / N  # np.random.rand(N)
    cmap = plt.cm.get_cmap('jet')
    # bars = ax.bars(theta, radii, width=width, bottom=0.0)

    ax.plot(theta, radii, color='black', ls='-', linewidth=0.5)
    ax.fill(theta, radii, '0.5', alpha=.3)
    # bars.set_alpha(0.5)
    # for r, bar in zip(radii, bars):
    #    print("burh")
    #    print(r)
    #    print(bar)
    #    #bar.set_facecolor(cm.jet_r(-r / (inc/4)))
    #    bar.set_alpha(0.5)

    show()

    # ERROR PLOTTING
    poles = []
    angles = np.arange(0.0, 2 * np.pi, 2 * np.pi / 360)
    for angle in angles:
        #print(angle)
        #poles.append(mic_loc(np.cos(angle) * 5, np.sin(angle) * 5, 0))
        rs = mic_sub(mic_loc(np.cos(angle) * 5, np.sin(angle) * 5, 0), np.array([0, 0, 0]))
        rs_sph = cart2sph(rs[0], rs[1], rs[2])
        #print(rs_sph)
        poles.append(rs_sph)
    sources = np.array(poles)
    
    error_all = []
    for rss in sources:
        L = L_multipole(ord, a, k, mics)
        D = D_multipole(n, a, f, rss, k, mics)
        A = A_multipole(L, D, n)
        # print(A)
        presmulti = pressure_withA(n, a, k, A)
        #print("PRES:")
        #print(presmulti)
        #pres = pressure(n, a, f, rss, k, rp_p_sph, rq_p_sph, rpq_p_sph, rqp_p_sph)
        err = Ynm_err(n, f, rss[1], rss[2], mic_test, presmulti, k, a)
        print(err)
        error_all.append(err)

    min = np.amin(error_all)
    max = np.amax(error_all)
    error_all = error_all

    #PLOT
    fig = figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    #ax.set_rorigin(min)
    inc = abs(min - max)
    print("inc")
    print(inc)
    N = 360
    ax.set_ylim(min, max + inc/8)
    theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
    radii = np.array(error_all)#+abs(min)  # 10 * np.random.rand(N)
    width = 2 * np.pi / N  # np.random.rand(N)
    cmap = plt.cm.get_cmap('jet')
    #bars = ax.bars(theta, radii, width=width, bottom=0.0)

    ax.plot(theta, radii, color='black', ls='-', linewidth=0.5)
    ax.fill(theta, radii, '0.5', alpha=.3)
    #bars.set_alpha(0.5)
    #for r, bar in zip(radii, bars):
    #    print("burh")
    #    print(r)
    #    print(bar)
    #    #bar.set_facecolor(cm.jet_r(-r / (inc/4)))
    #    bar.set_alpha(0.5)

    show()
    """


