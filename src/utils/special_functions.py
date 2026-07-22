import numpy as np

from scipy.special import jve, hankel1e


def j0(z):
    """Exponentially scaled Bessel function"""
    val = np.sqrt(np.pi) / np.sqrt(2 * z) * jve(0.5, z)
    return np.exp(abs(np.imag(z))) * val


def h0(z):
    """Exponentially scaled Hankel function"""
    val = np.sqrt(np.pi) / np.sqrt(2 * z) * hankel1e(0.5, z)
    return np.exp(1j * z) * val


def helmholtz_greens_function_3d(omega, r):
    """Helhholtz Green's function in 3D"""
    return 1j * omega * h0(omega * r) / (4 * np.pi)
