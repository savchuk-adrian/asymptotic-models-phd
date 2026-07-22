import numpy as np
import bempp.api

from utils.cq_solver import cq, cq_potential, cq_potential_plane
from utils.geometry import select_geometry, get_grid
from utils.special_functions import helmholtz_greens_function_3d
from config.incident_fields import define_incident_field
from config.settings import SimulationConfigTD, SimulationConfigPlaneTD


class BornSolver:
    def __init__(self, config: SimulationConfigTD | SimulationConfigPlaneTD):
        self.config = config
        self.M = len(config.centers)
        self.N = int(config.T / config.dt)
        self.rho = (1e-12) ** (0.5 / (self.N + 1))

        self.t = np.linspace(0, self.config.T, self.N + 1)

        self.grid_list = None
        self.equilibriums = None
        self.capacitances = None

    def initialize_geometry(self):
        """Precomputes grids and capacitances."""
        self.grid_list = select_geometry(
            self.config.geometry,
            self.config.centers,
            directions=getattr(self.config, "directions", None),
            angles=getattr(self.config, "angles", None),
        )
        self.equilibriums = self._compute_equilibriums()
        self.capacitances = self._compute_capacitances()

    def _ensure_initialized(self):
        """Internal helper to make sure geometry is ready before calculations."""
        if self.capacitances is None:
            self.initialize_geometry()

    def _compute_equilibriums(self):
        val = np.empty(self.M, dtype=object)
        for k, (grid_k) in enumerate(self.grid_list):
            space_k = bempp.api.function_space(grid_k, "DP", 0)
            slp_0 = bempp.api.operators.boundary.laplace.single_layer(
                space_k, space_k, space_k
            )
            one_grid_fun = bempp.api.GridFunction.from_ones(space_k)
            sigma_k, _ = bempp.api.linalg.gmres(slp_0, one_grid_fun, tol=1e-12)
            val[k] = bempp.api.GridFunction(space_k, coefficients=sigma_k.coefficients)
        return val

    def _compute_capacitances(self):
        val = np.empty(self.M, dtype=float)
        for k, sigma_k in enumerate(self.equilibriums):
            val[k] = sigma_k.integrate()
        return val

    def rhs(self, epsilon):
        self._ensure_initialized()
        val = np.empty((self.N + 1, self.M), dtype=complex)
        u_inc = define_incident_field(**self.config.inc_field)

        for k, (cap_k, c_k) in enumerate(zip(self.capacitances, self.config.centers)):
            val[:, k] = u_inc(c_k[0], c_k[1], c_k[2], self.t) * cap_k
        return epsilon * val

    def galerkin_matrix(self, omega, epsilon):
        self._ensure_initialized()
        """computing the diagonal matrix of capacitances"""
        return epsilon * np.diag(self.capacitances)

    def density(self, epsilon):
        self._ensure_initialized()
        G = self.rhs(epsilon)
        A = lambda s: self.galerkin_matrix(s, epsilon)
        val = cq(A, G, self.config.dt, self.rho, self.N, self.config.multistep_method)
        return val

    def potential(self, cap, epsilon, omega, c):
        x_0 = self.config.x_0.reshape(1, 3)
        c = np.array(c)
        dist = np.linalg.norm(x_0 - c)
        val = epsilon * helmholtz_greens_function_3d(omega, dist) * cap
        return val

    def compute_scattered_field(self, epsilon):
        self._ensure_initialized()
        J = self.density(epsilon)
        val = 0
        for k, (c_k, cap_k) in enumerate(zip(self.config.centers, self.capacitances)):
            A = lambda s: self.potential(cap_k, epsilon, s, c_k)
            val += cq_potential(
                A,
                J[:, k],
                self.config.dt,
                self.rho,
                self.N,
                self.config.multistep_method,
            )
        return val

    def potential_plane(self, cap, epsilon, omega, c, points, idx):
        val = np.zeros((points.shape[0], 1), dtype=np.complex128)
        X = points[idx, :]
        c = np.array(c).reshape(1, -1)
        dist = np.linalg.norm(X - c, axis=1).reshape(-1, 1)
        dist[dist == 0] = 1e-10
        G = helmholtz_greens_function_3d(omega, dist)
        res = epsilon * cap * G
        val[idx, 0] = res.ravel()
        return val

    def compute_scattered_field_plane(self, epsilon):
        self._ensure_initialized()

        xmin, xmax, ymin, ymax = self.config.limits

        points, idx = get_grid(
            self.config.geometry,
            self.config.centers,
            self.config.N_grid,
            xmin,
            xmax,
            ymin,
            ymax,
            epsilon=epsilon,
            plane=self.config.plane,
        )
        points = points.T  # (N_grid^2, 3)
        J = self.density(epsilon)  # (N, M)
        val = np.zeros(
            (self.N + 1, self.config.N_grid, self.config.N_grid), dtype=complex
        )
        for k, (c_k, cap_k) in enumerate(zip(self.config.centers, self.capacitances)):
            A = lambda s: self.potential_plane(cap_k, epsilon, s, c_k, points, idx)
            val += cq_potential_plane(
                A,
                J[:, k],
                self.config.dt,
                self.rho,
                self.N,
                self.config.multistep_method,
                self.config.N_grid,
            )
        mask_2d = idx.reshape(self.config.N_grid, self.config.N_grid)
        val[:, ~mask_2d] = np.nan
        return val
