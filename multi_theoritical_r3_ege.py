import Microphone as mc
import scipy.special as sp
from scipy.special import sph_harm, lpmv, spherical_jn, spherical_yn
import numpy as np
import recursion_r7 as recur
from numpy import matlib as mtlb
from scipy import linalg as LA
from pandas import DataFrame
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt5Agg')

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
            pnm = A * 4 * np.pi * (1j)**n * b[n] * np.conj(sparg_sph_harm(m, n, phis, thetas))
            mvec.append(pnm)
    return np.array(mvec), freq

def cart2sphr(xyz): # Ege'
    """ converting a multiple row vector matrix from cartesian to spherical coordinates """
    c = np.zeros((xyz.shape[0],2))
    r = []
    tt = []
    for i in range(xyz.shape[0]):
        
        
        x       = xyz[i][0]
        y       = xyz[i][1]
        z       = xyz[i][2]
        
        rad       =  np.sqrt(x*x + y*y + z*z)
        tt.append(rad)
        theta   =  np.arccos(z/rad)
        phi     =  np.arctan2(y,x)
        c[i][0] = theta
        c[i][1] = phi    
        r.append(rad)   
        
    return [c, np.array(r)]

def cart2sph_single(xyz): #(ege)
    """ converting a single row vector from cartesian to spherical coordinates """
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arccos(z/r)
    phi     =  np.arctan2(y,x)
    return(r, theta, phi)

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

def cart2sphr_sparg(xyz):
    """ converting a multiple row vector matrix from cartesian to spherical coordinates """
    c = np.zeros((xyz.shape[0],2))
    r = []
    tt = []
    for i in range(xyz.shape[0]):
        
        
        x       = xyz[i][0]
        y       = xyz[i][1]
        z       = xyz[i][2]
        
        rad       =  np.sqrt(x*x + y*y + z*z)
        tt.append(rad)
        theta   =  np.arccos(z/rad)
        phi     =  np.arctan2(y,x)
        c[i][0] = theta
        c[i][1] = phi    
        r.append(rad)   
    return(np.array(r), c[:,0], c[:,1])
    # return [c, np.array(r)]

def sph2cart(r, th, ph):
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    u = np.array([x, y, z])
    return u

def shd_add_noise(n,m):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    Ynm = np.random.rand(32,)* 1j
    for ind in range(32):
        tq = ths[ind]
        pq = phs[ind]
        Ynm[ind] = sparg_sph_harm(m, n, pq, tq)
    return Ynm

def mic_loc(x,y,z):
    return np.array([x, y, z])


def mic_sub(mic_p, mic_q):
    mic_pq = mic_p - mic_q
    return mic_pq


def S_nm(n, m, k = 1, r = 1 , th = np.pi/2, ph = np.pi/2):
    return (sp.spherical_jn(n, k * r)+1j * sp.spherical_yn(n, k * r)) * sparg_sph_harm(m, n, ph, th)

def R_nm(n, m, k = 1, r = 1 , th = np.pi/2, ph = np.pi/2):
    return sp.spherical_jn(n, k * r) * sparg_sph_harm(m, n, ph, th)


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
        Ynm = sparg_sph_harm(m, n, pq, tq) # 2025 - rafaely yazıyodu ama yanlış. çift condon var. gumerov bu.
        pnm += wq * cq * np.conj(Ynm)
    return pnm

def discorthonormality(N):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    CFnm = []
    for n in range(N+1):
        for m in range(-n, n+1):
            tmp = 0j
            for q in range(32):
                tmp += sparg_sph_harm(m, n, phs[q], ths[q]) * np.conj(sparg_sph_harm(m, n, phs[q], ths[q]))
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
        jnp = sp.spherical_jn(n, k * a, derivative=True)
        for m in range(-n, n+1):
            pnm = shd_nm(channels, n, m) * jnp * k * a**2 / (-rho*c)
            Pnm.append(pnm)
    return np.array(Pnm) * discorthonormality(Nmax)

def Y_nm(n, m):
    em32 = mc.EigenmikeEM32()
    estr = em32.returnAsStruct()
    ths = estr['thetas']
    phs = estr['phis']
    Ynm = np.zeros(32)*1j
    for ind in range(32):
        tq = ths[ind]
        pq = phs[ind]
        Ynm[ind] = sparg_sph_harm(m, n, pq, tq)
    return Ynm

def sparg_sph_harm(m, n, phi, theta):
    sph = (-1)**m * np.sqrt((2*n+1)/(4*np.pi) * sp.factorial(n-np.abs(m)) / sp.factorial(n+np.abs(m))) * sp.lpmn(int(np.abs(m)), int(n), np.cos(theta))[0][-1][-1] * np.exp(1j*m*phi)
    return sph

def sparg_sph_harm_list(m, n, phi, theta):
    s = []
    for i in range(len(phi)):
        ph = phi[i]
        th = theta[i]        
        sph = (-1)**m * np.sqrt((2*n+1)/(4*np.pi) * sp.factorial(n-np.abs(m)) / sp.factorial(n+np.abs(m))) * sp.lpmn(int(np.abs(m)), int(n), np.cos(th))[0][-1][-1] * np.exp(1j*m*ph)
        s.append(sph)    
    return (np.array(s))

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

def C_in_func(N, rp_p, s, freq, rot=0): #it was "C_in" (ege)
    """ s = rsrc_sph"""
    rho = 1.225
    c = 343
    t_size = (N+1)**2
    C_input= np.zeros(t_size)*1j
    k = 2 * np.pi * freq / c
    src_cart = sph2cart(1, s[1], s[2])
    k_vec = k*src_cart

    src_inward_sph = cart2sph(-src_cart[0],-src_cart[1],-src_cart[2]) #r,th,phi
    
    rp_p_cart = sph2cart(rp_p[0], rp_p[1], rp_p[2])
    phase = np.exp(-np.sum(k_vec*rp_p_cart)*1j)
    phase = 1
    for n in range(N+1):
        for m in range(-n, n+1):
            Ynm_gum = sparg_sph_harm(m, n, src_inward_sph[2], src_inward_sph[1]) #gumerov
            Ynm_raf = sph_harm(m, n, src_inward_sph[2], src_inward_sph[1]) # rafaely = scipy
            t = (n+1)**2 - (n-m)
            anm = np.conj(Ynm_gum)
            # Ccoef = anm / Ynm_raf
            # print("phase:\n ", phase)
            Cnm = anm * 4 * np.pi * (1j)**n * phase / (1j * 2 * np.pi * freq * -rho)
            C_input[t-1] = Cnm
    return C_input


def ynm_conj_approx(Cin,n, freq, s, rp_p):
    
    c = 343    
    k = 2 * np.pi * freq / c
    src_cart = sph2cart(1, s[1], s[2])
    k_vec = k*src_cart
    rp_p_cart = sph2cart(rp_p[0], rp_p[1], rp_p[2])
    phase = np.exp(-np.sum(k_vec*rp_p_cart)*1j)
    
    nlist = []
    rho = 1.225
    for i in range(n+1):
        for j in range(-i,i+1):
            nlist.append(1j**i)      
    ynm_approx = Cin*(1j * 2 * np.pi * freq * -rho) / (4*np.pi*np.array(nlist)*phase)
    return(ynm_approx)    
    
    
def C_in_new(N, rq_p, rq, r, s, freq, rot=0): #it was "C_in" (ege)
    """ s = rsrc_sph"""
    rho = 1.225
    c = 343
    t_size = (N+1)**2
    C_input= np.zeros(t_size)*1j
    k = 2 * np.pi * freq / c
    src_cart = sph2cart(1, s[1], s[2])
    k_vec = k*src_cart

    src_inward_sph = cart2sph(-src_cart[0],-src_cart[1],-src_cart[2]) #r,th,phi
    rq_p_cart = sph2cart(rq_p[0], rq_p[1], rq_p[2])
    phase = np.exp(-np.sum(k_vec*rq_p_cart)*1j)

    rq, rq_teta, rq_phi = cart2sph_single(rq)
    rr, r_teta, r_phi  = cart2sph_single(r)

    for n in range(N+1):
        for m in range(-n, n+1):
            Ynm_s = sparg_sph_harm(m, n, src_inward_sph[2], src_inward_sph[1])
            t = (n+1)**2 - (n-m)
            anm = np.conj(Ynm_s)*np.exp(-1j*rot*m)

            jn_kr = sp.spherical_jn(n, k * r)
            jn_krq = sp.spherical_jn(n, k * rq)

            Yr = sparg_sph_harm(m, n, r_phi, r_teta)
            Yq = sparg_sph_harm(m, n, rq_phi, rq_teta)

            Cnm = anm * 4 * np.pi * (1j)**n * phase * Yr * jn_kr / (Yq * jn_krq *1j * 2 * np.pi * freq * -rho)

            # C_input[t-1] = Cnm
    return Cnm



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
    L = np.eye(sqr_size)

    s = (n + 1) ** 2

    jhlp_fin = mtlb.repmat(jhlp, s, 1)
    SR = recur.reexp_coef(k, Lmax, Nmax, rpq_p_sph)
    L = SR.copy() * jhlp_fin
    return L

def get_key(poles, val):
    for key, value in poles.items():
        if val[0] == value[0] and val[1] == value[1] and val[2] == value[2]:
            k = key
            return k

def L_multipole(ordd, a, k, poles):
    """

    :param deg: Spherical harmonic order = 2
    :param a: radius of sphere = 0.042
    :param k: wave number
    :param poles: Coordinates of multipoles = mic locations
    :return: Reexpension coefficient (SR) multipoles
    """
    # poles=mics
    sqr_size = (ordd+1)**2 #(Lmax - 1) ** 2
    key, values = zip(*poles.items())
    Lsize = max(key)
    L_matrix = np.eye(sqr_size*Lsize, dtype=complex)

    for row in key:
        for col in key:
            if row == col:
                L = np.eye(sqr_size)
            else:
                rq_p = poles.get(row)
                rp_p = poles.get(col)
                rpq_p = mic_sub(rq_p, rp_p)
                rpq_p_sph = cart2sph(rpq_p[0], rpq_p[1], rpq_p[2])
                L = L_dipole(ordd, a, k, rpq_p_sph)                
            # print("row,col,L", row,col,L)
            L_matrix[((row-1) * sqr_size):((row) * sqr_size), ((col-1) * sqr_size):((col) * sqr_size)] = L
    # print("COND:",np.linalg.cond(L_matrix))
    return L_matrix


def D_multipole(C, mics, n, k, a):
    """
    C to D
    """
    size = (n+1)**2
    jhnp = []
    for i in range(n + 1):
        jnp = sp.spherical_jn(i, k * a, derivative=True)
        ynp = sp.spherical_yn(i, k * a, derivative=True)
        hnp = jnp + 1j * ynp
        for i in range((i) * 2 + 1):
            jhnp.append(jnp / hnp)
    jhnp = np.array(jhnp)
    
    jhnp_resized = np.resize(jhnp, size*len(mics))
    D_flat = C * -jhnp_resized
    return D_flat

def C_multipole(n, freq, s_sph, k, poles, rot):
    """
    :param n: Spherical harmonics order
    :param a: Radius of sphere
    :param freq: frequency
    :param s_sph: source spherical coordinates = rsrc_sph
    :param k: wave number
    :param poles: multipoles' spherical coordinates
    :return: D vector
    """
    key, values = zip(*poles.items())
    C_flat = np.array([])

    for keys in itertools.product(values):
        q = keys[0]
        rq_p =  q
        rq_p_sph = cart2sph(rq_p[0], rq_p[1], rq_p[2])
        C = C_in_func(n, rq_p_sph, s_sph, freq, rot)
        C_flat = np.append(C_flat, C)
    return C_flat

def C_multipole_new(n, freq, s_sph, k, poles, rq, r, rot):

    key, values = zip(*poles.items())
    C_flat = np.array([])

    for keys in itertools.product(values):
        q = keys[0]
        rq_p =  q
        # rq_p_sph = cart2sph(rq_p[0], rq_p[1], rq_p[2])
        # C = C_in_func(n, rq_p_sph, s_sph, freq, rot)
        C = C_in_new(n, rq_p, rq, r, s_sph, freq, rot=0)
        C_flat = np.append(C_flat, C)

    return C_flat


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
                Ynm = sparg_sph_harm(m, n, pq, tq)
                t = (n+1)**2 - (n-m) - 1
                potential += Anm[t]*Ynm / (jnp) #sph_harm(m, n, pw, tw) #Gumerov, Eq. 18
        pressure = -potential * c * rho / (k * a**2)
        mic32[ind] = pressure
    return mic32

def pressure_withA_multipole(n_low, a, k, Anm_all, no_of_rsmas):
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
    for arr in range(no_of_rsmas):
        Anm = Anm_all[arr*size:arr*size+size]
        mic32 = np.zeros(32) * 1j
        for ind in range(32):
            tq = ths[ind]
            pq = phs[ind]
            potential = 0
            for n in range(0, n_low+1):
                jnp = sp.spherical_jn(n, k * a, derivative=True)
                for m in range(-n,n+1):
                    Ynm = sparg_sph_harm(m, n, pq, tq)
                    t = (n+1)**2 - (n-m) - 1
                    potential += Anm[t] * Ynm / (jnp)  #sph_harm(m, n, pw, tw) #Gumerov, Eq. 30
            pressure = -potential * c * rho / (k * (a**2))
            mic32[ind] = pressure
        pressure_all.append(mic32)
    return pressure_all

def pressure_to_Anm(pressure, n_max, no_of_rsmas, k, a):
    Anm_scatter = []
    for arr in range(no_of_rsmas):
        pressure_temp = pressure[arr]
        Anm_scat_temp = np.array(shd_all2(pressure_temp, n_max, k, a)).flatten()
        Anm_scatter.append(Anm_scat_temp)
    Anm_scatter = np.array(Anm_scatter).flatten()
    return Anm_scatter

def Anm_to_D(Anm, L):
    D = np.dot(L, Anm)
    return D

def D_to_Cin(D,mics,jhnp, n):    
    size = (n+1)**2
    jhnp_resized = np.resize(jhnp, size*len(mics))
    C_in_scat = D * (1 / -jhnp_resized)
    # C_in_scat = D * -1
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
            Ynm.append(np.conj(sparg_sph_harm(m, n, phi_s, th_s))*phase)
    return np.array(Ynm)

def C_in_error(C_in, C_tilde):
    err = np.linalg.norm(C_in[0:9] - C_tilde[0:9]) / np.linalg.norm(C_in[0:9])
    #print(10*np.log(err**2))
    error = 10*np.log10(err**2)
    return error

def C_tilde_to_Anm(C, f, rho, mics):
    w = 2*np.pi*f
    block = int(len(C)/len(mics))
    n = int(np.sqrt(block)-1)
    # print("lenvlock", len(mics),block,n)
    C_split = C.reshape(len(mics),block)    
    anm = []    
    for row in range(len(C_split)):
        C_single = C_split[row]
        for n in range(0, n+1):
            for m in range(-n, n+1):
                anm.append(C_single*-1j*w*rho/(4*np.pi*((1j)**n)))
    anm = np.array([anm]).flatten()               
    return(anm)


def pfield_from_A(f, k, mesh_row, N, A_in, a):
    """
    Pressure field from SHD coefficients A_nm (still velocity-potential type).
    Mirrors the formula in pressure_withA_multipole.
    """
    rho, c = 1.225, 343.0
    mesh_sph, r = cart2sphr(mesh_row)
    th, ph = mesh_sph[:, 0], mesh_sph[:, 1]

    pr = np.zeros_like(r, dtype=complex)
    count = 0
    scale = -rho * c / (k * a**2)                # same factor as before

    for n in range(N + 1):
        jn_prime = spherical_jn(n, k * a, derivative=True)
        h_n      = spherical_jn(n, k * r) + 1j * spherical_yn(n, k * r)

        for m in range(-n, n + 1):
            Ynm = sparg_sph_harm_list(m, n, ph, th)
            pr += A_in[count] * h_n * Ynm
            count += 1

    return pr            # complex array; take .real if you want only the real part


def pfield_sphsparg(f, k, mesh_row, N, C_in):
    """ extrapolated pressure at distance r with given anm's [as a list]"""
    
    mesh_sph, r = cart2sphr(mesh_row)
    th = mesh_sph[:,0]
    ph = mesh_sph[:,1]
    rho = 1.225
    pr = 0
    kr = k*r
    w = 2*np.pi*f
            
    count = 0
    pr = 0
    for n in range(N+1):
        for m in range(-n,n+1): 
            term = C_in[count]*(-1j*w*rho)*spherical_jn(n, kr, derivative=False)*sparg_sph_harm_list(m, n, ph, th)
            pr = pr + term
            count += 1            
    return(pr)

def pfield_sphsparg_point(f, k, mesh_row, N, C_in):
    """ extrapolated pressure at distance r with given anm's [as a list]"""
    
    r, th, ph = cart2sph_single(mesh_row)

    rho = 1.225
    pr = 0
    kr = k*r
    w = 2*np.pi*f
            
    count = 0
    pr = 0
    for n in range(N+1):
        for m in range(-n,n+1): 
            term = C_in[count]*(-1j*w*rho)*spherical_jn(n, kr, derivative=False)*sparg_sph_harm(m, n, ph, th)
            pr = pr + term
            count += 1            
    return(pr)

def plot_contour(pressure, x, vsize, vmin=None, vmax=None, shift=None):
    """ contour plot (for pressure or angular error) with a 2D meshgrid """
    fig, ax = plt.subplots(figsize=(7, 7))
    if shift == None:
        shift = 0

    l = int(np.sqrt(len(pressure)))
    pressure_shaped = pressure.reshape(l, -1)
    pressure_real = pressure_shaped.real

    r_xx, r_yy = np.mgrid[-x+shift:x+shift:(vsize*1j), -x:x:(vsize*1j)]
    t = plt.contourf(r_xx, r_yy, pressure_real, cmap="jet", vmin=vmin, vmax=vmax)  # Set vmin and vmax
    ax.set_aspect("equal")
    p2 = ax.get_position().get_points().flatten()
    ax_cbar1 = fig.add_axes([p2[0], p2[2], p2[2]-p2[0], 0.025])
    print(p2)
    cbar = plt.colorbar(t, cax=ax_cbar1, orientation="horizontal", ticklocation='top')
    cbar.set_ticks(np.linspace(vmin, vmax, num=5))  # Or any number of ticks
    return ax

def rotvec(phis, N, Q):
    q = []
    for mic in range(Q):
        for n in range(N+1):
            for m in range(-n, n+1):
                q.append(np.exp(1j*m*phis[mic]))
    return np.array(q)

def rotatemat(MPmat, qrot):
    return qrot @ MPmat 

if __name__=='__main__':
    n = 3 #Spherical Harmonic Order
    # a = 42e-3   #Radius of sphere
    a = 8e-2   #Radius of sphere
    f = 2000    #Freq
    c = 343     #Speed of sound
    k = 2*np.pi*f/c #wave number
    order = n  # Spherical Harmonic Order for L matrix
    # 
    jhnp = []
    for i in range(n + 1):
        jnp = sp.spherical_jn(i, k * a, derivative=True)
        ynp = sp.spherical_yn(i, k * a, derivative=True)
        hnp = jnp + 1j * ynp
        for i in range((i) * 2 + 1):
            jhnp.append(jnp / hnp)
    jhnp = np.array(jhnp)
    
    source = mic_loc(1, 1, 0)
    rs = source # mic_sub(source, np.array([0, 0, 0]))
    rsrc = source # mic_sub(source, np.array([0, 0, 0])) # source coordinates
    rsrc_sph = cart2sph(rs[0], rs[1], rs[2]) # source coordinates in spherical coors.

    """ SCENE SETUP """
    center_x, center_y = 1., 1.
    # mics = {1: mic_loc(0.5, 0, 1), 2: mic_loc(1.5, 0.0, -1), 3: mic_loc(-0.5, 0, 3)}
    mics = {1: mic_loc(-0.2, 0, 0), 2: mic_loc(0.2, 0, 0)}
    p,q = mics.get(1), mics.get(2)
    middle = 0.5*(p+q)
    source_distance = np.linalg.norm(middle-source)
    print("distance btw. source and mics' middle point:", source_distance)
    key, values = zip(*mics.items())
    ad,bd,cd = list(zip(*values))
    
    no_of_rsmas = max(key) # = 3

    ar,br,cr = list(zip(rsrc))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Correctly set up 3D projection
    ax.scatter(ad, bd, cd, s=100, label='Microphone Locations')
    ax.scatter(ar, br, cr, c="r", s=250, label='Source Location')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Scatter Plot of Microphones and Source')
    ax.legend()
    plt.show()

    size = (n + 1) ** 2

    """ Step 1 """
    C_in = C_multipole(n, f, rsrc_sph, k, mics, rot=0)
    D = D_multipole(C_in, mics, n, k, a)
    
    """ Step 2 """
    """ L_multipole returns: Reexpansion coefficient (SR) multipoles" """
    L = L_multipole(order, a, k, mics) # ord = 2 = spherical Harmonic Order for L matrix   
    _, Anm_all = A_multipole(L, D, n)
    
    """ Step 3  """
    presmulti = pressure_withA_multipole(n, a, k, Anm_all, no_of_rsmas)
    Anm_tilde = pressure_to_Anm(presmulti, n, no_of_rsmas, k, a)
    D_tilde = Anm_to_D(Anm_tilde, L)
    C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)

    """ Step 3.5 Rotation """
    """
    Anm_tilde_diag = np.diag(Anm_tilde)
    o = (n + 1) ** 2
    p1 = []
    p2 = []
    qr = rotvec([-np.pi/4, np.pi/4], n, 2)
    # qr = rotvec([0, 0], n, 2)
    Anm_tilde_rot = rotatemat(Anm_tilde_diag, qr)
    
    D_tilde_rot = Anm_to_D(Anm_tilde_rot, L)
    C_in_tilde_rot = D_to_Cin(D_tilde_rot, mics, jhnp, n)
    c1_tilde_rot = C_in_tilde_rot[0:o]
    c2_tilde_rot = C_in_tilde_rot[o:2*o]

    # Step 4: rotation iteration 
    err = []
    p_rot = np.linspace(0, 2 * np.pi, 360)
    
    for rot in p_rot:
        qr = rotvec([rot, 0], n, 2)
        Ar = rotatemat(Anm_tilde_diag, qr)
        Anm_tilde_rot = Ar
        D_tilde = Anm_to_D(Anm_tilde_rot, L)
        C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)
        
        c1_tilde = C_in_tilde[0:o]
        c2_tilde = C_in_tilde[o:2*o]
        
        # c1_tilde = C_in[0:o]  # trying original C_in instead of C_in_tilde
        # c2_tilde = C_in[o:2*o]
        
        pressure_p1 = pfield_sphsparg_point(f, k, 0.5*(p+q) - p, n, c1_tilde)
        pressure_p2 = pfield_sphsparg_point(f, k, 0.5*(p+q) - q, n, c2_tilde)
        p1.append(pressure_p1)
        p2.append(pressure_p2)
        
        # err.append(abs(pressure_p1-pressure_p2))
        err.append(np.linalg.norm(c1_tilde-c2_tilde))

        
    err_ = np.array(err)
    ind = err.index(min(err))
    print("min hatanın açısı [derece]", np.degrees(p_rot[ind]))
    
    print("order:", n)
    print("max error:", np.max(err_))
    print("min error:", np.min(err_))
    
    fig = plt.figure()
    plt.plot(p_rot, err)
    plt.title('Error vs Rotation Angle')
    plt.xlabel('Rotation Angle (radians)')
    plt.ylabel('Error')
    plt.grid()
    plt.show()
    """

    def mesh_row_shift(vsize=50, cm=0.55, shift=0):
        # if shift < 0:
        #     r_x, r_y, r_z = np.mgrid[-cm-shift:cm:(vsize * 1j), -cm:cm:(vsize * 1j), 0:0:1j]
        #     print("grid:", -cm-shift, "to", cm)
        # else:
        #     r_x, r_y, r_z = np.mgrid[-cm:cm:(vsize * 1j), -cm-shift:cm-shift:(vsize * 1j), 0:0:1j]
        #     print("grid:", cm, "to", cm-shift)
        r_x, r_y, r_z = np.mgrid[-cm - shift:cm - shift:(vsize * 1j), -cm:cm:(vsize * 1j), 0:0:1j]
        print("grid:", -cm - shift, "to", cm - shift)
        mesh_row = np.stack((r_x.ravel(), r_y.ravel(), r_z.ravel()), axis=1)
        return(mesh_row)


    def mesh_row_center(center_x, cm=0.55, vsize=50):
        """Square 2-D mesh centred on `center_x` in the global frame."""
        r_x, r_y, r_z = np.mgrid[
                        center_x - cm: center_x + cm: (vsize * 1j),
                        -cm:  cm: (vsize * 1j),
                        0:  0: 1j
                        ]
        return np.stack((r_x.ravel(), r_y.ravel(), r_z.ravel()), axis=1)


    """ Step 4: Testing Cin """
    """ mesh grid for measurement points """
    rc = 0.5  # extrapolation range in meters
    cm = rc + 0.05  # side of counter map's square in meter
    vsize = 50  # number of pressure calculations along one side for contour plot
    r_x, r_y, r_z = np.mgrid[-cm:cm:(vsize * 1j), -cm:cm:(vsize * 1j), 0:0:1j]
    mesh_row = np.stack((r_x.ravel(), r_y.ravel(), r_z.ravel()), axis=1)

    global_mesh = mesh_row_center(0.0)  # one grid for plotting
    rel_mesh_p = global_mesh - p  # r  relative to sphere 1
    rel_mesh_q = global_mesh - q  # r  relative to sphere 2


    o = (n + 1) ** 2  # order = n, block sizes

    a1_tilde = Anm_tilde[0:o]
    a2_tilde = Anm_tilde[o:2 * o]

    pressure_a1 = pfield_from_A(f, k, rel_mesh_p, n, a1_tilde, a)
    pressure_a2 = pfield_from_A(f, k, rel_mesh_q, n, a2_tilde, a)

    scattered = pressure_a1 + pressure_a2

    c1_tilde = C_in_tilde[0:o]
    c2_tilde = C_in_tilde[o:2*o]

    c1_tilde_orj = C_in[0:o]  # trying original C_in instead of C_in_tilde
    c2_tilde_orj = C_in[o:2*o]

    pressure_p1_cin = pfield_sphsparg(f, k, mesh_row, n, c1_tilde_orj)
    # pressure_p1_cin_tilde = pfield_sphsparg(f, k, mesh_row, n, c1_tilde)
    # pressure_p1_c1_tilde_rot = pfield_sphsparg(f, k, mesh_row, n, c1_tilde_rot)

    # pressure_p1_c2_tilde_rot = pfield_sphsparg(f, k, mesh_row, n, c2_tilde_rot)
    vmin, vmax = -1.5, 2  # Set colorbar limits

    # ax1 = plot_contour(pressure_p1_c2_tilde_rot, cm, vsize, vmin=vmin, vmax=vmax)  # contour plot
    # plt.title("pressure using c2_tilde_rot")

    ax2 = plot_contour(pressure_p1_cin, cm, vsize, vmin=vmin, vmax=vmax)  # contour plot
    plt.title("Pressure using c1_tilde_orj")

    # ax3 = plot_contour(pressure_p1_cin_tilde, cm, vsize, vmin=vmin, vmax=vmax)  # contour plot
    # plt.title("Pressure using c1_tilde")

    # ax4 = plot_contour(pressure_p1_c1_tilde_rot, cm, vsize, vmin=vmin, vmax=vmax)  # contour plot
    # plt.title("Pressure using c1_tilde_rot")

    # ax5 = plot_contour(pressure_a1_p1_tilde, cm, vsize, vmin=vmin, vmax=vmax, shift=p[0])  # contour plot
    # plt.title("scattered field a1_tilde")

    # ax6 = plot_contour(pressure_a2_p1_tilde, cm, vsize, vmin=vmin, vmax=vmax, shift=q[0])  # contour plot
    # plt.title("scattered field a2_tilde")

    total = scattered + pressure_p1_cin  # total field
    print(np.abs(scattered).min(), np.abs(scattered).max())
    print(np.abs(pressure_p1_cin).min(), np.abs(pressure_p1_cin).max())

    # scattered =  pressure_a1

    ax6 = plot_contour(total, cm, vsize, vmin=vmin, vmax=vmax)  # contour plot
    plt.title("total scattered field a1+a2 + incident field")

    plt.show()
    # pp5 = pressure_p1_cin_tilde.real
    # plt.imshow(pp5.reshape(50,50))

    # print("C-input")
    # print(C_in)
    # print("C-output")
    # print(C_in_tilde)
    # print("Diff")

    # C_diff = C_in - C_in_tilde
    # print(DataFrame(C_diff))
    # print("magnitude Diff")
    # print(DataFrame(np.linalg.norm(C_diff[i]) for i in range(len(C_diff))))
