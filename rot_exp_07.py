import numpy as np
from convertangle import cart2sph_single, cart2sphr, sph2cart
from scipy.special import sph_harm, lpmv, spherical_jn, spherical_yn
import matplotlib.pyplot as plt
import matplotlib as mpl
from sphere_scattering import *

""" pressure contours for NEW Cin (sadeleşmemiş Ynm'ler ile) and Cin_tilde's """

# 650,1300,2600
# 1,2,3

pi = np.pi

def a_nm(mt, mp, m, n):    
    a_nm_tilde = np.conj(sph_harm(m, n, mp, mt))    
    return(a_nm_tilde)

def a_nm_rot(mt, mp, m, n, azi_rot):    
    a_nm_tilde = np.conj(sph_harm(m, n, mp, mt))*np.exp(-1j*azi_rot*m)
    return(a_nm_tilde)

def grad(anm, r, teta, phi, k, N, rotation):
    """ pressure grad in spherical coordinates, to be used for obtaining velocity and then intensity """
    r_unit = np.array([np.sin(teta)*np.cos(phi), np.sin(teta)*np.sin(phi), np.cos(teta)])
    p_unit = np.array([-np.sin(phi)*np.sin(teta), np.sin(teta)*np.cos(phi), 0])
    t_unit = np.array([np.cos(teta)*np.cos(phi), np.cos(teta)*np.sin(phi), -np.sin(teta)])
    
    kr = k*r
    gradp = 0.
    count = 0
    for n in range(N+1):
        for m in range(-n,n+1):
            # a_nm = a_nm_rot(mt,mp,m,n,rotation)
            a_nm = anm[count]
            
            pr_der = a_nm*4*pi*((1j)**n)*k*spherical_jn(n, kr, derivative=True)*sph_harm(m, n, phi, teta)
            Y = sph_harm(m+1, n, phi, teta)
            if np.isnan(Y):               
                Y = 0.
                
            pteta_der = a_nm*4*pi*((1j)**n)*spherical_jn(n, kr, derivative=False)*(m*(1/np.tan(teta))*sph_harm(m, n, phi, teta) 
            + np.sqrt((n-m)*(n+m+1))*np.exp(-1j*phi)*Y)            
            
            pphi_der = a_nm*4*pi*((1j)**n)*spherical_jn(n, kr, derivative=False)*1j*m*sph_harm(m, n, phi, teta)
            grad_sum = pr_der*r_unit+(1/r)*pteta_der*t_unit + (1/(r*np.sin(teta)))*pphi_der*p_unit
            gradp = gradp + grad_sum
            
            count += 1
    return(gradp)

def calc_u(pressure_grad, ro, k):
    w = 2*pi*f
    u = -pressure_grad/(1j*w*ro)
    return(u)
        
def pressure_field_anm(k, r, teta, phi, N, anm_tilde):
    """ extrapolated pressure at distance r with given anm's [as a list]"""
    pr = 0
    kr = k*r
    count = 0
    for n in range(N+1):
        for m in range(-n,n+1): 
            term = anm_tilde[count]*4*np.pi*((1j)**n)*spherical_jn(n, kr, derivative=False)*sph_harm(m, n, phi, teta)
            pr = pr + term
            count += 1
    return(pr)

def pressure_field_multi(f, k, mesh_row, N, C_in):
    """ extrapolated pressure at distance r with given anm's [as a list]"""
    
    mesh_sph, r = cart2sphr(mesh_row)
    teta = mesh_sph[:,0]
    phi = mesh_sph[:,1]
    
    pr = 0
    kr = k*r
    count = 0
    w = 2*pi*f
    for n in range(N+1):
        for m in range(-n,n+1): 
            term = C_in[count]*(-1j*w*rho)*spherical_jn(n, kr, derivative=False)*sph_harm(m, n, phi, teta)
            pr = pr + term
            count += 1
    return(pr)

def ia(p, u):
    """ local active intensity with known pressure and velocity """
    I_a = 0.5*(p*np.conj(u)).real
    In = I_a/np.linalg.norm(I_a)
    return(I_a, In)
    
def interpolation(r,teta,phi,N,k,mt,mp,rotation):
    pressure_grad = []
    velocity = []
    pressure = []
    I_active = []
    I_n = []
    if np.isscalar(r):
        r = np.full((len(teta),1)  , r)
        
    for i in range(len(teta)):
        
        p_grad = grad(r[i], teta[i], phi[i], k, N,mt,mp,rotation) # grad of the pressure
        u = calc_u(p_grad, ro, k)  # local velocity calculation using euler momentum equation with pressure grad
        p = pressure_field_anm(mt, mp, k, r[i], teta[i] , phi[i], N)  # extrapolated pressure at r (due to monochromatic plane wave coming from mt,mp)

        I_act, In = ia(p, u)  # local active intensity
        
        if abs(I_act) < 1e-8:
            arg = 0.
        
        pressure_grad.append(p_grad)
        velocity.append(u)
        pressure.append(p)
        I_active.append(I_act)
        I_n.append(In)
    
    pressure_grad = np.array(pressure_grad)
    velocity = np.array(pressure_grad)
    pressure = np.array(pressure)
    I_active = np.array(I_active)
    
    I_n = np.array(I_n)
    If,Ig,Ih = list(zip(*I_active))

    return(pressure,I_active)

def interpolation_shifted(anm_tilde, mesh_row, N, k, rotation, shift_vector):
    
    mesh_sph, r = cart2sphr(mesh_row)
    teta = mesh_sph[:,0]
    phi = mesh_sph[:,1]

    pressure_grad = []
    velocity = []
    pressure = []
    I_active = []
    I_n = []
    if np.isscalar(r):
        r = np.full((len(teta),1)  , r)
        
    for i in range(len(teta)):
        
        p_grad = grad(anm_tilde, r[i], teta[i], phi[i], k, N,rotation) # grad of the pressure
        u = calc_u(p_grad, ro, k)  # local velocity calculation using euler momentum equation with pressure grad
        p = pressure_field_anm(k, r[i], teta[i] , phi[i], N, anm_tilde)*np.exp(-1j*np.sum(k*-shift_vector))  # extrapolated pressure at r (due to monochromatic plane wave coming from mt,mp)

        I_act, In = ia(p, u)  # local active intensity
        
        pressure_grad.append(p_grad)
        velocity.append(u)
        pressure.append(p)  
        I_active.append(I_act)
        I_n.append(In)
    
    pressure_grad = np.array(pressure_grad)
    velocity = np.array(pressure_grad)
    pressure = np.array(pressure)
    I_active = np.array(I_active)
    
    I_n = np.array(I_n)
    If,Ig,Ih = list(zip(*I_active))

    return(pressure,I_active)

def intensity_error(I_p,m_vector):
    angle_err = []
    d = []
    for i in range(len(I_p)):
        arg = np.sum(I_p[i]*-m_vector)/(np.linalg.norm(I_p[i])*np.linalg.norm(m_vector))
        
        if abs(arg) < 1e-8:
            arg = 0. 
        elif arg>1 and arg<1.05:   
            arg = 1.
        elif arg<-1 and arg>-1.05:   
            arg = -1.
        d.append(arg)    
        angle_err.append(np.degrees(np.arccos(arg))) 
        
    return(np.average(angle_err),angle_err,d)

def intensity_scale(I,sc=1):
    # I_norm = np.array([np.linalg.norm(I[i]) for i in range(len(I))])
    I_norm = np.array([np.linalg.norm(I)])
    I_scaled = np.array([I[i]/(sc*max(I_norm)) for i in range(len(I))])
    return(I_scaled)
    
    return
def plot_contour(pressure, x, vsize):
    """ contour plot (for pressure or angular error) with a 2d meshgrid """
    fig, ax = plt.subplots(figsize=(7, 7))
    
    l = int(np.sqrt(len(pressure)))
    pressure_shaped = pressure.reshape(l, -1)
    pressure_real = pressure_shaped.real
    
    r_xx,r_yy = np.mgrid[-x:x:(vsize*1j), -x:x:(vsize*1j)]
    t = plt.contourf(r_xx,r_yy, pressure_real)
    ax.set_aspect("equal")
    p2 = ax.get_position().get_points().flatten()
    ax_cbar1 = fig.add_axes([p2[0],p2[2], p2[2]-p2[0], 0.025])

    plt.colorbar(t,cax=ax_cbar1 ,orientation="horizontal",ticklocation = 'top')
#    plt.show()
    return(ax)
    
def plot_i_vectors(mesh_row, Iu,Iv,Iw, clr, ax):
    """ Intensity vectors as a quiver plot"""
    if ax==0:        
        ax = plt.gca()
    ax.quiver(mesh_row[:,0],mesh_row[:,1], Iu, Iv, scale = 1, color=clr, units='width')
    return ax

def error(I_p1_line_scaled, I_p2_line_scaled):    
    
    I_p1_ave = np.array([np.mean(I_p1_line_scaled[:,0]), np.mean(I_p1_line_scaled[:,1]), np.mean(I_p1_line_scaled[:,2])])
    I_p2_ave = np.array([np.mean(I_p2_line_scaled[:,0]), np.mean(I_p2_line_scaled[:,1]), np.mean(I_p2_line_scaled[:,2])])

    for i,val in enumerate(I_p1_ave):
        if abs(I_p1_ave[i])< 1e-8:
            I_p1_ave[i] = 0
    
    for i,val in enumerate(I_p2_ave):
        if abs(I_p2_ave[i])< 1e-8:
            I_p2_ave[i] = 0
        
    arg = np.sum(I_p1_ave*I_p2_ave)/(np.linalg.norm(I_p1_ave)*np.linalg.norm(I_p2_ave))
    angle_err = np.degrees(np.arccos(arg))
    return(angle_err)

def jhnp_func(n,k,a):
    jhnp = []
    for i in range(n + 1):
        jnp = sp.spherical_jn(i, k * a, derivative=True)
        ynp = sp.spherical_yn(i, k * a, derivative=True)
        hnp = jnp + 1j * ynp
        for i in range((i) * 2 + 1):
            jhnp.append(jnp / hnp)
    jhnp = np.array(jhnp)   
    return(jhnp)

def jhnp_(n,k,a):
    jhnp = []
    for ind in range(n + 1):
        jnp = sp.spherical_jn(ind, k * a, derivative=True)
        ynp = sp.spherical_yn(ind, k * a, derivative=True)
        hnp = jnp + 1j * ynp      
        for jnd in range(-ind, ind+1):            
            jhnp.append(jnp / hnp)
    jhnp = np.tile(jhnp,len(mics))
    return(jhnp)

def rotvec(phis, N, Q):
    q = []
    for mic in range(Q):
        for n in range(N+1):
            for m in range(-n, n+1):
                q.append(np.exp(1j*m*phis[mic]))
    return np.array(q)

def rotatemat(MPmat, qrot):
    return MPmat @ qrot

n = 2

a = 0.042
c = 343.0 
rho = 1.225 # ambient density
rc = 1  # extrapolation distance
f = 1500
k = 2*np.pi*f/c
order = n

source = np.array([1, 1, 0])
rs = source # mic_sub(source, np.array([0, 0, 0]))
rs_sph = cart2sph(rs[0], rs[1], rs[2]) # source coordinates in spherical coors.
kv = k*source/(np.linalg.norm(source))
m_vector = source
# mics = {1: np.array([0, 0, 1]), 2: np.array([0.2, 0, 1])}
mics = {1: mic_loc(0.1, 0, 0), 2: mic_loc(0.4, 0, 0)}
p,q = mics.get(1), mics.get(2)
middle = 0.5*(p+q)
source_distance = np.linalg.norm(middle-source)
print("distance btw. source and mics' middle point:", source_distance)
shift_vector = mic_sub(mics.get(2), mics.get(1))

""" source and mic locations plot """
key, values = zip(*mics.items())
ad,bd,cd = list(zip(*values))
no_of_poles = max(key) # = 3
ar,br,cr = list(zip(rs))
mx,my,mz = list(zip(middle))
fig=plt.figure()
ax = Axes3D(fig)    
ax.scatter(ad,bd,cd, s=100)
ax.scatter(ar,br,cr, c="r", s=250)
ax.scatter(mx,my,mz, c="y", s=150)

""" r grid """
cm = rc+0.05  # side of counter map's square in meter
vsize = 50 # kontür plotu için tek kenarda kaç adet intensity vektör hesaplanacağı

""" maybe used with contour plot's 2d meshgrid or error calculation """
r_x,r_y,r_z = np.mgrid[-cm:cm:(vsize*1j), -cm:cm:(vsize*1j), 0:0:1j]
mesh_row = np.stack((r_x.ravel(), r_y.ravel(), r_z.ravel() ), axis=1)

""" line points """

# xs = np.linspace(0.15, 0.15, 10)
# ys = np.linspace(-0.1, 0.1, 10)
# zs = np.linspace(0, 0, 10)
# line = np.array([xs,ys,zs]).T

# line_sph, line_radius = cart2sphr(line)
# teta_line = line_sph[:,0]
# phi_line = line_sph[:,1]

# line_p2 = np.array([-xs,ys,zs]).T
# line_p2_sph, line_p2_radius = cart2sphr(line_p2)
# teta_line_p2 = line_p2_sph[:,0]
# phi_line_p2 = line_p2_sph[:,1]

""" Step 1 """
jhnp = jhnp_(n,k,a)

# C_in = C_multipole(n, f, rs_sph, k, mics, rot=0)

for r in mesh_row:
    rq = r - q
    rp = r - p
    C_in = C_multipole_new(n, f, rs_sph, k, mics, rq, r, rot=0 )
    
    
    
    
    
    
    D = D_multipole(C_in, mics,n,k,a)
    """
    c1 = C_in[0:9]
    c2 = C_in[9:18]
    C_shifted = c1 * np.exp(-1j*np.sum(kv*shift_vector))
    """
    
    """ Step 2 """
    L = L_multipole(order, a, k, mics) # ord = 2 = spherical Harmonic Order for L matrix   
    _, Anm_all = A_multipole(L, D, n)
    
    """ Step 3  """
    presmulti = pressure_withA_multipole(n, a, k, Anm_all, len(mics))
    Anm_tilde = pressure_to_Anm(presmulti, n, len(mics),k,a)
    
    # Anm_tilde_diag = np.diag(Anm_tilde)
    # o = (n+1)**2
    # p1 = []
    # p2 = []
    # qr = rotvec([0, 0], n, 2)
    # Ar = rotatemat(Anm_tilde_diag, qr)
    # Anm_tilde_rot = Ar
    Anm_tilde_rot = Anm_tilde
    D_tilde = Anm_to_D(Anm_tilde_rot, L)
    C_in_tilde = D_to_Cin(D_tilde, mics, jhnp, n)

c1_tilde = C_in_tilde[0:o]
# c2_tilde = C_in_tilde[o:2*o]

c1_tilde_orj = C_in[0:o]  # trying original C_in instead of C_in_tilde
# c2_tilde = C_in[o:2*o]

"""
pressure_p1 = pressure_field_multi(f, k, mesh_row, n, c1_tilde)
pressure_p1_orj = pressure_field_multi(f, k, mesh_row, n, c1_tilde_orj)

ax2 = plot_contour(pressure_p1, cm, vsize)  # contour plot 
plt.title("pressure using c1_tilde")

ax3 = plot_contour(pressure_p1_orj, cm, vsize)  # contour plot 
plt.title("pressure using original c1")

"""
