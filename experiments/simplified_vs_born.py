import bempp.api
import numpy as np

from dataclasses import asdict

from asymptotic_solvers.time_domain.sgfl import EfficientGalerkinFoldyLax
from asymptotic_solvers.time_domain.born import BornSolver
from config.settings import SimulationConfigTD

bempp.api.DEFAULT_DEVICE_INTERFACE = 'numba'

def fibonacci_lattice(N, r):
    Phi = (1+np.sqrt(5))/2
    Phi_inv = Phi**(-1)
    indices = np.linspace(-N, N, 2*N+1)
    
    lat = np.empty(2*N+1, dtype=float)
    lon = np.empty(2*N+1, dtype=float)

    k=0
    for i in indices:
        lat[k] = np.arcsin(2*i/(2*N+1))
        lon[k] = 2 * np.pi * np.mod(i, Phi) * Phi_inv
        if lon[k] < -np.pi:
            lon[k] += 2 * np.pi
        if lon[k] > np.pi:
            lon[k] -= 2 * np.pi
        k += 1
    
    x = r * np.cos(lon) * np.cos(lat)
    y = r * np.sin(lon) * np.cos(lat)
    z = r * np.sin(lat)
    points = np.stack((x, y, z), axis=1)

    return  points

points = fibonacci_lattice(35, 1.5)

directions = [np.random.randn(3) for _ in range(points.shape[0])]
directions = directions/np.linalg.norm(directions, keepdims = True, axis=1, ord=2)
angles = [np.random.uniform(0, 2*np.pi) for _ in range(points.shape[0])]

CONFIG = SimulationConfigTD(
        centers=points,
        geometry='ellipsoid_cut',
        directions=directions,
        angles=angles,
        inc_field={
            'u_inc_name': 'modulated_gaussian',
            'd': np.array([1, -1, 1]) / np.sqrt(3),
            'omega': 2 * np.pi,
            'sigma': 100,
            'A': 3
        },
        T=11,
        dt=0.01,
        multistep_method='TTR',
        x_0=np.array([-1.5, -1.5, -1.5]).reshape(-1, 1)
    )

data = asdict(CONFIG)

model = EfficientGalerkinFoldyLax(CONFIG)
model_born = Born(CONFIG)

epsilon_list = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
err_list = []

for epsilon in epsilon_list:
    eps = str(epsilon).replace('.', '_')

    u_s = model.scattered_field(epsilon)
    u_B = model_born.scattered_field(epsilon)

    data[f'u_s_{eps}'] = u_s
    data[f'u_B_{eps}'] = u_B

    err = np.linalg.norm(u_s - u_B, ord=np.inf)
    err_list.append(err)

data['epsilon_list'] = epsilon_list
data['error_list'] = err_list

np.savez('sgfl_vs_born_fibonacci_lattice.npz', **data)