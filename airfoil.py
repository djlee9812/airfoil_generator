import matplotlib.pyplot as plt
import numpy as np
from math import comb
import seaborn as sns
sns.set_theme()

def naca4(max_camber, camber_loc, thickness, closed=False):
    """
    NACA 4 digit airfoil generator
    max_camber: maximum camber as percentage of chord
    camber_loc: distance of max camber from leading edge in c/10
    thickness: maximum thickness as percent of chord
    closed: close trailing edge
    """
    m = max_camber / 100
    p = camber_loc / 10
    t = thickness / 100
    n_pt = 100
    beta = np.linspace(0, np.pi, n_pt)
    x = (1 - np.cos(beta))/2
    a0 = 0.2969
    a1 = -0.126
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1036 if closed else -0.1015 
    y_t = 5 * t * (a0*np.sqrt(x) + a1*x + a2*x**2 + a3 *x**3 + a4*x**4)
    if m == 0:
        y_c = np.zeros(n_pt)
        dycdx = np.zeros(n_pt)
    elif p == 0:
        y_c = np.zeros(n_pt)
        dycdx = np.zeros(n_pt)
    else:
        y_c = np.piecewise(
            x, 
            [x <= p, x > p], 
            [lambda xc: m/p**2 * (2*p*xc - xc**2),
            lambda xc: m/(1-p)**2 * ((1-2*p) + 2*p*xc - xc**2)] 
        )
        dycdx = np.piecewise(
            x,
            [x <= p, x > p],
            [lambda xc: 2*m/p**2 * (p-xc),
            lambda xc: 2*m/(1 - p)**2 * (p-xc)]
        )
    theta = np.arctan(dycdx)
    x_u = x - y_t * np.sin(theta)
    x_l = x + y_t * np.sin(theta)
    y_u = y_c + y_t * np.cos(theta)
    y_l = y_c - y_t * np.cos(theta)

    airfoil = np.zeros((n_pt*2, 2))
    airfoil[:n_pt, 0] = np.flip(x_u)
    airfoil[:n_pt, 1] = np.flip(y_u)
    airfoil[n_pt:, 0] = x_l
    airfoil[n_pt:, 1] = y_l
    return airfoil

def naca5(opt_cl, camber_loc, camber_type, thickness, closed=False):
    """
    NACA 5 digit airfoil generator
    opt_cl: Designed Cl * 3/20
    camber_loc: distance of max camber from leading edge in c/20
    camber_type: 0 for normal camber, 1 for reflex camber
    thickness: maximum thickness as percent of chord
    closed: close trailing edge
    """
    n_pt = 100
    beta = np.linspace(0, np.pi, n_pt)
    x = (1 - np.cos(beta)) / 2
    a0 = 0.2969
    a1 = -0.126
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1036 if closed else -0.1015 
    y_t = 5 * thickness / 100 * (a0*np.sqrt(x) + a1*x + a2*x**2 + a3 *x**3 + a4*x**4)
    if camber_type == 0:
        if camber_loc == 1:
            r = 0.058
            k1 = 361.4
        elif camber_loc == 2:
            r = 0.126
            k1 = 51.64
        elif camber_loc == 3:
            r = 0.2025
            k1 = 15.957
        elif camber_loc == 4:
            r = 0.29
            k1 = 6.643
        elif camber_loc == 5:
            r = 0.391
            k1 = 3.23
        else:
            print("Invalid camber location")
            return
        y_c = np.piecewise(
            x,
            [x < r, x >= r],
            [lambda xc: k1/6 * (xc**3 - 3*r*xc**2 + r**2*(3 - r)*xc),
             lambda xc: k1 * r**3 / 6 * (1 - xc)]
        )
        dycdx = np.piecewise(
            x,
            [x < r, x >= r],
            [lambda xc: k1/6 * (3*xc**2 - 6*r*xc + r**2*(3 - r)),
             lambda xc: -k1 * r**3 / 6 ]
        )
        y_c *= opt_cl / 2
        dycdx *= opt_cl / 2
    elif camber_type == 1:
        if camber_loc == 2:
            r = 0.13
            k1 = 51.99
            k2 = k1 * 0.000764
        elif camber_loc == 3:
            r = 0.217
            k1 = 15.793
            k2 = k1 * 0.00677
        elif camber_loc == 4:
            r = 0.318
            k1 = 6.520
            k2 = k1 * 0.0303
        elif camber_loc == 5:
            r = 0.441
            k1 = 3.191
            k2 = k1 * 0.1355
        else:
            print("Invalid camber location")
            return
        y_c = np.piecewise(
            x,
            [x < r, x >= r],
            [lambda xc: k1/6 * ((xc-r)**3 - k2/k1*(1-r)**3*xc - r**3*xc + r**3),
             lambda xc: k1/6 * (k2/k1*(xc-r)**3 - k2/k1*(1-r)**3*xc - r**3*xc + r**3)]
        )
        dycdx = np.piecewise(
            x,
            [x < r, x >= r],
            [lambda xc: k1/6 * (3*(xc-r)**2 - k2/k1*(1-r)**3 - r**3),
             lambda xc: k1/6 * (3*k2/k1*(xc-r)**2 - k2/k1*(1-r)**3 - r**3) ]
        )
        y_c *= opt_cl / 2
        dycdx *= opt_cl / 2
    else:
        print("Invalid S value,", camber_type)
        return None
    theta = np.arctan(dycdx)
    x_u = x - y_t * np.sin(theta)
    x_l = x + y_t * np.sin(theta)
    y_u = y_c + y_t * np.cos(theta)
    y_l = y_c - y_t * np.cos(theta)

    airfoil = np.zeros((n_pt*2, 2))
    airfoil[:n_pt, 0] = np.flip(x_u)
    airfoil[:n_pt, 1] = np.flip(y_u)
    airfoil[n_pt:, 0] = x_l
    airfoil[n_pt:, 1] = y_l
    return airfoil

def kulfan(Aupper, Alower, dz=0, n_pt=100, twist=0):
    """Kulfan Airfoil Generator

    Args:
        Aupper (list[float]): Upper surface shape function weights
        Alower (list[float]): Lower surface shape function weights
        dz (float, optional): Trailing edge gap. Defaults to 0.
        n_pt (float, optional): Number of points for each top and bottom surface. Defaults to 100.

    Returns:
        np.2darray: Column vectors: [x, y] in (n_pt x 2)
    """
    wu = np.array(Aupper)
    wl = np.array(Alower)
    n = len(wu) - 1
    N1 = 0.5
    N2 = 1
    betas = np.linspace(0, np.pi, n_pt)
    x = (1 - np.cos(betas)) / 2
    C = (x ** N1) * ((1-x) ** N2)
    r = np.arange(n+1)
    K = np.array([comb(n, i) for i in r])
    Su = np.zeros(n_pt)
    Sl = np.zeros(n_pt)
    for i in range(n_pt):
        Su[i] = np.sum(wu * K * (x[i] ** r) * ((1-x[i]) ** (n-r)))
        Sl[i] = np.sum(wl * K * (x[i] ** r) * ((1-x[i]) ** (n-r)))

    y_u = C * Su + x * dz
    y_l = C * Sl - x * dz

    x_full = np.concatenate([np.flip(x), x])
    y_full = np.concatenate([np.flip(y_u), y_l])

    rho = np.sqrt(x_full**2 + y_full**2)
    phi = np.arctan2(y_full, x_full) - np.deg2rad(twist)

    x_twist = rho * np.cos(phi)
    y_twist = rho * np.sin(phi)

    airfoil = np.zeros((n_pt*2, 2))
    airfoil[:, 0] = x_twist
    airfoil[:, 1] = y_twist
    # airfoil[:n_pt, 0] = np.flip(x)
    # airfoil[:n_pt, 1] = np.flip(y_u)
    # airfoil[n_pt:, 0] = x
    # airfoil[n_pt:, 1] = y_l

    

    return airfoil

def plot_airfoil(x, y, axIn=None):
    """ Plot an airfoil given x and y coordinates

    Args:
        x (list[float]): X-coordinates of airfoil
        y (list[float]): Y-coordinates of airfoil
        axIn (plt.Axes, optional): plt axes to plot to. Defaults to None.
    """
    sns.set(font_scale=1.3)
    sns.set_style('ticks')
    if axIn is None:
        plt.figure()
        ax = plt.subplot(111)
    else: 
        ax = axIn
    ln = ax.plot(x, y)
    ax.set_xlabel(r'$x/c$')
    ax.set_ylabel(r'$y/c$')
    plt.xlim(-0.1, 1.1)
    plt.ylim(np.min(y)*1.2, np.max(y)*1.2)
    ax.set_aspect('equal')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    if axIn is not None:
        return ln
    plt.show()

def write_dat_file(coords, filename="airfoil.dat", airfoilName=''):
    fmt = ['%.8f\t', '%.8f']
    np.savetxt(filename, coords, fmt=fmt, header=airfoilName)

def view_airfoil_file(filename):
    """ Plot MSES airfoil files

    Args:
        filename (string): Path to MSES .airfoil file
    """
    f = open(filename, 'r')
    blade_text = f.read()
    f.close()

    bladelines = blade_text.split("\n")
    bladelines = bladelines[2:-2]
    xcoords = []
    ycoords = []
    for line in bladelines:
        nums = line.split()
        xcoords.append(float(nums[0]))
        ycoords.append(float(nums[1]))

    plot_airfoil(xcoords, ycoords)

if __name__ == "__main__":
    pass