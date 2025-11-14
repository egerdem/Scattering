import numpy as np
import Microphone as mc
from sphere_scattering import *


n = 2  #Spherical Harmonic Order
N_max = n
a = 42e-3   #Radius of sphere
f = 5000    #Freq
c = 343     #Speed of sound
k = 2*np.pi*f/c #wave number
ord = 2   # Spherical Harmonic Order for L matrix

em32 = mc.EigenmikeEM32()
estr = em32.returnAsStruct()
wts = estr['weights']
ths = estr['thetas']
phs = estr['phis']
    
source = mic_loc(3, 3, 0 )
rs = source # mic_sub(source, np.array([0, 0, 0]))
rsrc = source # mic_sub(source, np.array([0, 0, 0])) # source coordinates
rsrc_sph = cart2sph(rs[0], rs[1], rs[2]) # source coordinates in spherical coors.

center_x, center_y = 1., 1.
mics = {1: mic_loc(0.5, 0, 1), 2: mic_loc(1.5, 0.0, -1), 3: mic_loc(-0.5, 0, 3)}
key, values = zip(*mics.items())
no_of_poles = max(key) # = 3

size = (n+1)**2
""" Step 1 """
D, jhnp, C_in = D_multipole(n, a, f, rsrc_sph, k, mics)
jhnp = np.resize(jhnp, size*len(mics))

""" Step 2 """
""" L_multipole returns: Reexpansion coefficient (SR) multipoles" """
L = L_multipole(ord, a, k, mics) # ord = 2 = spherical Harmonic Order for L matrix   
_, Anm_all = A_multipole(L, D, n)

""" Step 3  """
# pres_single = pressure_withA(n, a, k, A)
presmulti = pressure_withA_multipole(n, a, k, Anm_all, no_of_poles)
Anm_scatter = []
for arr in range(no_of_poles):
    pressure_temp = presmulti[arr]
    channels = pressure_temp
    Pnm = []
    rho = 1.225
    c = 343
    for n in range(N_max+1):
        jnp = sp.spherical_jn(n, k * a, derivative=True)
        for m in range(-n, n+1):
            pnm_1 = np.zeros(np.shape(channels[0]))*1j
            for ind in range(32):
                cq = channels[ind]
                wq = wts[ind]
                tq = ths[ind]
                pq = phs[ind]
                Ynm = sph_harm(m, n, pq, tq) #Rafaely Ynm
                pnm_1 += wq * cq * np.conj(Ynm)                   
            pnm = pnm_1 * jnp * k * a**2 / (-rho*c)
            Pnm.append(pnm)
            
    out = np.array(Pnm) * discorthonormality(N_max)
    
    Anm_scat_temp = np.array(out).flatten()
    Anm_scatter.append(Anm_scat_temp)        
    
Anm_scatter = np.array(Anm_scatter).flatten()

Anm_tilde = Anm_scatter
