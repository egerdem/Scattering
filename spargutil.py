
# utility functions for plotting and frequently used list-dict-array manipulation
# first version: 04.08.22

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import json
from scipy.spatial.transform import Rotation as R
from itertools import combinations, permutations
import pickle
from collections import defaultdict
from scipy.io import wavfile

def dump(file_name, variable):
    file = open(file_name, 'wb')
    pickle.dump(variable, file)
    file.close()
    print(f"{file_name=} is dumped")
    return

def load(file_name):
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    print("%s is loaded " % file_name)
    return data

def list2dict(ls):
    mic_dict_z = {}
    for i in range(len(ls)):
        mic_dict_z[i + 1] = ls[i]
    return mic_dict_z

def json2dict(filename):
    f = open(filename)
    return json.load(f)

def json2miclist_z(filename):
    f = open(filename)
    jdict = json.load(f)
    mic_dict = {}
    mic_dict_z = {}
    mic_list = []
    for key, value in jdict.items():
        if key[0:3]=="Pos":
            mic_dict[key] = value
            mic_list.append(value)
    mic_list_z = np.c_[mic_list, np.zeros(len(mic_dict))]
    for i in range(len(mic_list_z)):
        mic_dict_z[i] = mic_list_z[i]
    return mic_list_z, mic_dict_z

def add_axes(ax):
    ax.quiver(0, 0, 0,  # <-- starting point of vector
              0, 0, 1.5,  # <-- directions of vector
              color='blue', alpha=.2, lw=3, )

    ax.quiver(0, 0, 0,  # <-- starting point of vector
              0, 1.5, 0,  # <-- directions of vector
              color='blue', alpha=.2, lw=3, )

    ax.quiver(0, 0, 0,  # <-- starting point of vector
              1.5, 0, 0,  # <-- directions of vector
              color='blue', alpha=.2, lw=3, )

    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$', fontsize=20)
    ax.set_zlabel('$Z$', fontsize=20)

def set_aspect_equal(ax):
    """
    Fix the 3D graph to have similar scale on all the axes.
    Call this after you do all the plot3D, but before show
    """
    X = ax.get_xlim3d()
    Y = ax.get_ylim3d()
    Z = ax.get_zlim3d()
    a = [X[1]-X[0],Y[1]-Y[0],Z[1]-Z[0]]
    b = np.amax(a)
    ax.set_xlim3d(X[0]-(b-a[0])/2,X[1]+(b-a[0])/2)
    ax.set_ylim3d(Y[0]-(b-a[1])/2,Y[1]+(b-a[1])/2)
    ax.set_zlim3d(Z[0]-(b-a[2])/2,Z[1]+(b-a[2])/2)
    ax.set_box_aspect(aspect = (1,1,1))

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_scene(src, mics):
    key, mic_locs = zip(*mics.items())
    ad, bd, cd = list(zip(*mic_locs))
    ar, br, cr = list(zip(src))
    fig = plt.figure()
    ax50 = Axes3D(fig)
    ax50.scatter(ad, bd, cd, s=100)
    ax50.scatter(ar, br, cr, c="r", s=250)
    return

def plot_scene_with_ax(src, mics, ax):
    if type(mics).__name__ == "dict":
        key, mic_locs = zip(*mics.items())
        ad, bd, cd = list(zip(*mic_locs))
    else:
        ad, bd, cd = mics[:, 0], mics[:, 1], mics[:, 2]
    ar, br, cr = list(zip(src))
    ax.scatter(ad, bd, cd, s=100)
    ax.scatter(ar, br, cr, c ="g", s=200)
    return

def plot_scatter_multip_init(mics):
    figt = plt.figure()
    ax = Axes3D(figt, auto_add_to_figure=False)
    ax.scatter(mics[:,0], mics[:,1], mics[:,2], s=100)
    figt.add_axes(ax)
    return ax

def plot_scatter_multip(mics, ax, c="C0", s=100):
    if type(mics).__name__ == "dict":
        key, mic_locs = zip(*mics.items())
        ad, bd, cd = list(zip(*mic_locs))
    else:
        ad, bd, cd = list(zip(*mics))
    ax.scatter(ad, bd, cd, c=c,s=s)
    return

def plot_scatter_singlep(mic):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.scatter(mic[0], mic[1], mic[2], s=100)
    fig.add_axes(ax)
    return

def plot_scatter_singlep_with_ax(mic, ax, c="y", s=20):
    ax.scatter(mic[0], mic[1], mic[2], c=c, s=s)
    return
def pairings(l1, l2):
    # https://math.stackexchange.com/questions/3813903/notation-help-set-of-pairings-of-items-from-two-sets
    num_elements = min(len(l1), len(l2))
    l1_combs = list(combinations(l1, num_elements))
    l2_perms = list(permutations(l2, num_elements))
    ls = []
    count = 1
    err_list = []
    # pair_list = []
    pairdict = defaultdict()
    for c in l1_combs:
        for p in l2_perms:
            err = np.sum(np.sqrt((np.subtract(p, c) ** 2)))
            err_list.append(err)
            # pair_list.append(pair)
            pair = tuple(zip(p, c))
            pairdict[pair] = err
            count += 1

    val = np.min(err_list)
    index = np.argmin(err_list)
    key = [k for k, v in pairdict.items() if v == np.min(err_list)][0]
    print(key)
    print("index:", index)
    print("val:", val)
    return key

def vnorm(v):
    nrm = np.linalg.norm(v)
    nv = v / nrm
    return nv, nrm

def angle_btwn_2_vecs(a, b):
    # atan2((Va x Vb).Vn, Va.Vb)
    b, _ = vnorm(b)
    a, _ = vnorm(a)

    t = np.arccos(np.sum(a * b))
    print("arccos =", np.degrees(t))

    Vn = np.array([0,0,1])  # plane normal = z axis
    t1 = np.dot(np.cross(a, b),Vn)
    t2 = np.arctan2(t1, np.dot(a, b))
    print("arctan2 =", np.degrees(t2))

    return (np.degrees(t), np.degrees(t2))

def rotate_geo(mic_list_z, srcind, refind, front):
    """ src2mic1 vektörünü global x ekseni ile align ettiğimiz fonksiyon
    hangi vektörü align etmek istiyorsak ona göre refind değişebilir
    refind: bu vektör hangi mikrofonun konumu kullanılarak yapılıyorsa o mikrofonun mic_list_z'deki indexi
    srcind : index of the mic_list_z that correspond to the source
    """
    # atan2((Va x Vb).Vn, Va.Vb)
    # arctan (t2) işareti de dikkate alınmak için yapıldı, henüz detaylı kontrol edilmedi, o yüzden döndürülmüyor.
    front = vnorm(front)[0]
    Vn = np.array([0,0,1])  # plane normal = z axis

    mic = mic_list_z[refind]
    src = mic_list_z[srcind]
    src2mic1 = src - mic
    src2mic1, _ = vnorm(src2mic1)

    t1 = np.dot(np.cross(src2mic1, front),Vn)
    t2 = np.arctan2(t1, np.dot(src2mic1, front))
    print("arctan2, t2=", np.degrees(t2))

    t = np.arccos(np.sum(src2mic1 * front))
    print("arccos, t=", np.degrees(t))

    new_list = mic_list_z.copy()
    r = R.from_rotvec([0, 0, -t2])
    rotz = r.as_matrix()
    rotated_mic_list_z = np.matmul(new_list, rotz)
    rotated_mic_dict_z = list2dict(rotated_mic_list_z)
    print("all microphones and sources have rotated %f degrees" %np.degrees(t2))
    return (rotated_mic_list_z, rotated_mic_dict_z)

def maximize(func, a, b, centre, v, srf_list,gridvecs, Nmax, mposlist, precision):
    ite = 1
    """
    Find the point in the given range which maximises the input function with an
    error at most equal to `precision` / 2.
    The fuction to be maximized should either strictly increasing and then
    strictly decreasing or vice versa.
    """
    (left, right) = (a, b)
    while True:
        if right - left <= precision:
            return (left + right) / 2
        # print("iter no.", ite)
        ite += 1
        left_third = ((2 * left) + right) / 3
        right_third = (left + (2 * right)) / 3

        if func(left_third,centre, v, srf_list, gridvecs, Nmax, mposlist) < func(right_third,centre, v, srf_list, gridvecs, Nmax, mposlist):
            (left, right) = (left_third, right)
        else:
            (left, right) = (left, right_third)

def write_wav(filename ='Sine1.wav', Fs=48000, freq = 1500, length = 2):

    # CREATE SINE WAVE AND SAVE AS .WAV
    t = np.linspace(0, length, Fs * length)  # Produces a (value of length) second Audio-File
    y = np.sin(freq * 2 * np.pi * t)  # Has frequency of sine wave
    wavfile.write(filename, Fs, y)

if __name__=='__main__':
    print("This script includes utility functions for frequently used list-dict-array manipulation + plotting")
    # front = np.array([1, 0, 0])
    # rotated_mic_list_z , rotated_mic_dict_z = rotate_geo(mic_list_z, 3, 0, front)

    # l1 = [175, 26, 30, 65, 20]
    # l2 = [13, 31, 72, 168, 191]
    #
    # l1 = [175, 26, 30, 65, 480]
    # l2 = [14, 31, 72, 168, 191, 500, 174]
    #
    # # print("starting")
    # t = pairings(l1,l2)


    front = np.array([1, 0, 0])
    # mic_list_z, _ = json2miclist_z("./data/mocap2/twosourcemocap.json")
    mic_list_z, _ = json2miclist_z("./data/largehexalab.json")

    mic_list_z[0:8, 2] = 1.5

    srcind, refind = 7, 6

    rotation_line1 = mic_list_z[refind]
    rotation_line2 = mic_list_z[srcind]
    rotation_line_vec = rotation_line2 - rotation_line1
    rotated_mic_list_z, rotated_mic_dict_z = rotate_geo(mic_list_z, srcind, refind, front)

    rotated_line1 = rotated_mic_list_z[refind]
    rotated_line2 = rotated_mic_list_z[srcind]
    rotated_line_vec = rotated_line2 - rotated_line1

    x_values = [rotation_line1[0], rotation_line2[0]]
    y_values = [rotation_line1[1], rotation_line2[1]]
    z_values = [rotation_line1[2], rotation_line2[2]]

    x = [rotated_line1[0], rotated_line2[0]]
    y = [rotated_line1[1], rotated_line2[1]]
    z = [rotated_line1[2], rotated_line2[2]]

    fig = plt.figure(figsize=plt.figaspect(1))  # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1, 1, 1])
    plot_scatter_multip(mic_list_z, ax, c="k")
    ax.plot(x_values, y_values, z_values, '-ok')
    ax.set_ylim3d(-1,1)
    ax.set_xlim3d(-1,1)
    add_axes(ax)
    plt.show()

    fig = plt.figure(figsize=plt.figaspect(1))  # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1, 1, 1])
    plot_scatter_multip(rotated_mic_list_z, ax, c="r")
    ax.plot(x, y, z, '-ok')
    ax.set_ylim3d(-1,1)
    ax.set_xlim3d(-1,1)
    add_axes(ax)
    plt.show()

