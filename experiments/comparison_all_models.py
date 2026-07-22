import bempp.api
import numpy as np

from dataclasses import asdict

from asymptotic_solvers.time_domain.gfl import GalerkinFoldyLaxSolver
from asymptotic_solvers.time_domain.sgfl import SimplifiedGalerkinFoldyLaxSolver
from asymptotic_solvers.time_domain.born import BornSolver
from bem_solver.time_domain.bem import BemSolver
from config.settings import SimulationConfigTD

bempp.api.DEFAULT_DEVICE_INTERFACE = "numba"

geo_config = np.load("geometry_config.npz", allow_pickle=True)

CONFIG = SimulationConfigTD(
    # centers=[(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (0, 0, 0)],
    centers=geo_config["centers"],
    geometry="ellipsoid_cut",
    directions=geo_config["directions"],
    angles=geo_config["angles"],
    inc_field={
        "u_inc_name": "modulated_gaussian",
        "d": np.array([1, -1, 1]) / np.sqrt(3),
        "omega": 2 * np.pi,
        "sigma": 3,
        "A": 3,
    },
    T=10,
    dt=0.05,
    multistep_method="TTR",
    x_0=np.array([-1, -1, -1]).reshape(-1, 1),
)

data = asdict(CONFIG)

model_bem = BemSolver(CONFIG)
model_gfl = GalerkinFoldyLaxSolver(CONFIG)
model_sgfl = SimplifiedGalerkinFoldyLaxSolver(CONFIG)
model_born = BornSolver(CONFIG)

epsilon_list = np.array([0.5])
data["epsilon_list"] = epsilon_list

for epsilon in epsilon_list:
    eps = str(epsilon).replace(".", "_")
    u_gfl = model_gfl.compute_scattered_field(epsilon)
    u_sgfl = model_sgfl.compute_scattered_field(epsilon)
    u_born = model_born.compute_scattered_field(epsilon)
    u_bem = model_bem.compute_scattered_field(epsilon)

    data[f"u_gfl_{eps}"] = u_gfl
    data[f"u_sgfl_{eps}"] = u_sgfl
    data[f"u_born_{eps}"] = u_born
    data[f"u_bem_{eps}"] = u_bem


# import bempp.api
# import numpy as np

# from dataclasses import asdict

# from asymptotic_solvers.gfl_td import GalerkinFoldyLaxSolver
# from asymptotic_solvers.sgfl_td import SimplifiedGalerkinFoldyLaxSolver
# from asymptotic_solvers.born import BornSolver
# from bem_solver.bem_td import BemSolver
# from config.settings import SimulationConfigTD

# bempp.api.DEFAULT_DEVICE_INTERFACE = 'numba'

# CONFIG = SimulationConfigTD(
#     centers=[(-1, 0, 0), (1, 0, 0)],
#     geometry='ellipsoid',
#     directions=None,
#     angles=None,
#     inc_field={
#         'u_inc_name': 'modulated_gaussian',
#         'd': np.array([1, 0, 0]),
#         'omega': 2*np.pi,
#         'sigma': 3,
#         'A': 3
#     },
#     T=1,
#     dt=0.5,
#     multistep_method='TTR',
#     x_0 = np.array([0, 0, 0]).reshape(-1, 1)
# )

# model_bem = BemSolver(CONFIG)
# model_gfl = GalerkinFoldyLaxSolver(CONFIG)
# model_sgfl = SimplifiedGalerkinFoldyLaxSolver(CONFIG)
# model_born = BornSolver(CONFIG)

# epsilon = 0.1

# u_gfl = model_gfl.compute_scattered_field(epsilon)
# u_sgfl = model_sgfl.compute_scattered_field(epsilon)
# u_born = model_born.compute_scattered_field(epsilon)
# u_bem = model_bem.compute_scattered_field(epsilon)
