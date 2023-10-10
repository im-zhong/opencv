import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import color, io


def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    return 1. / (1. + norm(grad(x))**2)


img = io.imread('/home/zhangzhong/src/opencv/imgs/orange.jpg')
img = color.rgb2gray(img)
img = img - np.mean(img)

sigma = 0
# Smooth the image to reduce noise and separation between noise and edge becomes clear
img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)

F = stopping_fun(img_smooth)


def default_phi(x):
    # Initialize surface phi at the border (5px from the border) of the image
    # i.e. 1 outside the curve, and -1 inside the curve
    phi = np.ones(x.shape[:2])
    phi[5:-5, 5:-5] = -1.
    return phi


dt = 1.

for i in range(n_iter):
    dphi = grad(phi)
    dphi_norm = norm(dphi)

    dphi_t = F * dphi_norm

    phi = phi + dt * dphi_t


def curvature(f):
    fy, fx = grad(f)
    norm = np.sqrt(fx**2 + fy**2)
    Nx = fx / (norm + 1e-8)
    Ny = fy / (norm + 1e-8)
    return div(Nx, Ny)


def div(fx, fy):
    fyy, fyx = grad(fy)
    fxy, fxx = grad(fx)
    return fxx + fyy


def dot(x, y, axis=0):
    return np.sum(x * y, axis=axis)


v = 1.
dt = 1.

g = stopping_fun(img_smooth, alpha)
dg = grad(g)

for i in range(n_iter):
    dphi = grad(phi)
    dphi_norm = norm(dphi)
    kappa = curvature(phi)

    smoothing = g * kappa * dphi_norm
    balloon = g * dphi_norm * v
    attachment = dot(dphi, dg)

    dphi_t = smoothing + balloon + attachment

    phi = phi + dt * dphi_t
