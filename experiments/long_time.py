import bempp.api
import numpy as np

from dataclasses import asdict

from asymptotic_solvers.time_domain.gfl import GalerkinFoldyLaxSolver
from asymptotic_solvers.time_domain.sgfl import SimplifiedGalerkinFoldyLaxSolver
from asymptotic_solvers.time_domain.born import BornSolver
from bem_solver.time_domain.bem import BemSolver
from config.settings import SimulationConfigTD

bempp.api.DEFAULT_DEVICE_INTERFACE = "numba"

CONFIG = SimulationConfigTD(
    centers=[(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
    geometry="ellipsoid_cut",
    inc_field={
        "u_inc_name": "plane_wave_sigmoid_TD",
        "d": np.array([1, -1, 1]) / np.sqrt(3),
        "omega": 2 * np.pi,
        "sigma": 2,
        "A": 5,
    },
    T=25,
    dt=0.05,
    multistep_method="TTR",
    x_0=np.array([0, 0, 0]).reshape(-1, 1),
)


data = asdict(CONFIG)

model = GalerkinFoldyLax(CONFIG)
bem = BEM(CONFIG)

epsilon_list = np.array([0.05])
data["epsilon_list"] = epsilon_list

for epsilon in epsilon_list:
    eps = str(epsilon).replace(".", "_")
    u_bem = bem.scattered_field(epsilon)
    # u_G = model.scattered_field(epsilon)

    data[f"u_BEM_{eps}"] = u_bem
    # data[f'u_G_{eps}'] = u_G
