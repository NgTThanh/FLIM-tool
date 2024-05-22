import numpy as np
from .sofism_lib import process_measurement_joblib
from scipy.optimize import curve_fit

path = "E:\FLIM\pakiet_studencki\calib_35nm_red\Run132"


def gaussian(X, a, xo, yo, sx, sy, t, c):
    x, y = X
    p1 = 0.5 * (np.cos(t) / sx) ** 2 + 0.5 * (np.sin(t) / sy) ** 2
    p2 = -np.sin(2 * t) / (4 * sx ** 2) + np.sin(2 * t) / (4 * sy ** 2)
    p3 = np.sin(t) ** 2 / (2 * sx ** 2) + np.cos(t) ** 2 / (2 * sy ** 2)
    return (a * np.exp(-(p1 * (x - xo) ** 2 + 2 * p2 * (x - xo) * (y - yo) + p3 * (y - yo) ** 2)) + c).ravel()


def fit_gaussian(data):
    dx, dy = data.shape
    X = np.mgrid[0:dx:1, 0:dy:1]
    # data = gaussian_filter(data, sigma=2)
    idx0 = np.where(data == data.max())
    xo, yo = idx0[0][0], idx0[1][0]
    dr = data.max() - data.min()
    p0 = [data.max(), xo, yo, dx / 5, dx / 5, 0, data.min()]
    bds = ([data.min(), xo - dx / 5, yo - dy / 5, 0, data.min()],
           [2 * data.max(), xo + dx / 5, yo + dy / 5, dx, data.max()])

    try:
        popt, pcov = curve_fit(gaussian, X, data.ravel(), p0=p0)
    except RuntimeError:
        return (-1, -1, -1, dx, -1, -1, -1)
    return popt


npa = lambda x: np.array(x)


def calc_shifts(data):
    shifts = np.zeros((23, 2))
    sigmas_x = []
    sigmas_y = []
    g_params = []
    for i in range(23):
        a, xo, yo, sx, sy, t, c = fit_gaussian(data[i])
        g_params.append((a, xo, yo, sx, c))
        sigmas_x.append(sx)
        sigmas_y.append(sy)
        if i == 11:
            shifts[i, 0] = xo
            shifts[i, 1] = yo
        shifts[i, 0] = xo
        shifts[i, 1] = yo
    return shifts, npa(sigmas_x), npa(sigmas_y)

def get_shifts(path):
    data = process_measurement_joblib(path, 1, 5, 5)
    s, sx, sy = calc_shifts(data.sum(axis=-1))
    print(sx)
    print(sy)
    np.save(path + "/red_shifts.npy", s)
    return path + "/red_shifts.npy"   