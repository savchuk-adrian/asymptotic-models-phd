import numpy as np
import bempp.api

from bempp.api.assembly.blocked_operator import BlockedOperator

from utils.geometry import select_geometry
from utils.cq_solver import cq, cq_potential
from config.incident_fields import define_incident_field
from config.settings import SimulationConfigTD


class BemSolver:
    def __init__(self, config: SimulationConfigTD):
        self.config = config
        self.M = len(self.config.centers)
        self.N = int(self.config.T / self.config.dt)
        self.rho = (1e-12) ** (0.5 / (self.N + 1))

        self.t = np.linspace(0, self.config.T, self.N + 1)

        self.grids = None
        self.merged_grid = None
        self.N_dof = None
        self.merged_space = None
        self.spaces = None

    def _ensure_initialized(self):
        """Internal helper to make sure geometry is ready before calculations."""
        if self.grids is None:
            self.initialize_geometry()

    def initialize_geometry(self):
        """Precomputes grids and capacitances."""
        self.grids = select_geometry(
            self.config.geometry,
            self.config.centers,
            directions=getattr(self.config, "directions", None),
            angles=getattr(self.config, "angles", None),
        )
        self.spaces = [bempp.api.function_space(g, "DP", 0) for g in self.grids]

        self.merged_grid = bempp.api.grid.union(self.grids)
        self.merged_space = bempp.api.function_space(self.merged_grid, "DP", 0)
        self.N_dof = self.merged_grid.number_of_elements

    def rhs(self, epsilon):
        self._ensure_initialized()

        u_inc = define_incident_field(**self.config.inc_field)
        val = np.empty((self.N + 1, self.N_dof), dtype=complex)
        for n, t_n in enumerate(self.t):
            current_col = 0
            for k, (grid_k, c_k) in enumerate(zip(self.grids, self.config.centers)):
                N_k = grid_k.number_of_elements
                col = slice(current_col, current_col + N_k)

                @bempp.api.complex_callable
                def second_member(x, n, domain_index, result):
                    result[0] = u_inc(
                        epsilon * (x[0] - c_k[0]) + c_k[0],
                        epsilon * (x[1] - c_k[1]) + c_k[1],
                        epsilon * (x[2] - c_k[2]) + c_k[2],
                        t_n,
                    )

                g_dt_k = bempp.api.GridFunction(self.spaces[k], fun=second_member)
                val[n, col] = g_dt_k.projections(self.spaces[k])
                current_col += N_k
        return val

    def galerkin_matrix(self, omega, epsilon):
        val = BlockedOperator(self.M, self.M)
        for k, (space_k, c_k) in enumerate(zip(self.spaces, self.config.centers)):
            for l, (space_l, c_l) in enumerate(zip(self.spaces, self.config.centers)):
                slp_omega_kl = bempp.api.operators.boundary.helmholtz.single_layer(
                    space_k, space_l, space_l, omega, epsilon, c_k, c_l
                )
                val[l, k] = slp_omega_kl
        return val

    def density(self, epsilon):
        G = self.rhs(epsilon)
        A = lambda s: self.galerkin_matrix(s, epsilon).weak_form().A
        val = cq(A, G, self.config.dt, self.rho, self.N, self.config.multistep_method)
        return val

    def potential(self, density, epsilon, omega, c):
        space = density.space
        slp_pot_scaled = bempp.api.operators.potential.helmholtz.single_layer_scaled(
            space, self.config.x_0, omega, epsilon, c
        )
        return slp_pot_scaled.evaluate(density)

    def compute_scattered_field(self, epsilon):
        J = self.density(epsilon)
        val = np.zeros((self.N + 1), dtype=complex)
        current_col = 0
        for space_k, c_k in zip(self.spaces, self.config.centers):
            N_k = space_k.grid.number_of_elements
            col = slice(current_col, current_col + N_k)
            J_k = J[:, col]
            sigma_k = lambda G: bempp.api.GridFunction(space_k, coefficients=G)
            A = lambda s, G: self.potential(sigma_k(G), epsilon, s, c_k)
            val += cq_potential(
                A, J_k, self.config.dt, self.rho, self.N, self.config.multistep_method
            )
            current_col += N_k
        return val
