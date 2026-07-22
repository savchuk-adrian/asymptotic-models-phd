import numpy as np
import bempp.api

from app.utils.geometry import select_geometry, merge_and_save_meshes
from app.utils.cq_solver import cq, cq_potential_BEM
from app.config.incident_fields import define_incident_field

eps_m = 10 ** (-12)


class BEM:
    def __init__(self, parameters=None):
        self.geometry = parameters.geometry
        self.center_list = parameters.centers
        self.M = len(self.center_list)

        self.dt = parameters.dt
        self.T = parameters.T
        self.multistep_method = parameters.multistep_method
        self.N = int(self.T / self.dt)
        self.t = np.linspace(0, self.T, self.N + 1)
        self.rho = eps_m ** (0.5 / (self.N + 1))

        self.u_inc = parameters.inc_field
        self.x_0 = parameters.x_0

        if parameters.directions is None:
            self.grid_list = select_geometry(self.geometry, self.center_list)
        else:
            self.grid_list = select_geometry(
                self.geometry,
                self.center_list,
                directions=parameters.directions,
                angles=parameters.angles,
            )

        self.grid_merged = merge_and_save_meshes(self.grid_list)
        self.N_dof = self.grid_merged.number_of_elements
        self.space_merged = bempp.api.function_space(self.grid_merged, "DP", 0)
        self.space_list = self.compute_space_list()

    def compute_space_list(self):
        val = np.empty(self.M, dtype=object)
        for k, (grid_k) in enumerate(self.grid_list):
            space_k = bempp.api.function_space(grid_k, "DP", 0)
            val[k] = space_k
        return val

    def RHS(self, epsilon):
        u_inc = define_incident_field(**self.u_inc)
        val = np.empty((self.N + 1, self.N_dof), dtype=complex)
        for n, t_n in enumerate(self.t):
            current_col = 0
            for k, (grid_k, c_k) in enumerate(zip(self.grid_list, self.center_list)):
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

                g_dt_k = bempp.api.GridFunction(self.space_list[k], fun=second_member)
                val[n, col] = g_dt_k.projections(self.space_list[k])
                current_col += N_k
        return epsilon**2 * val

    def A_Galerkin(self, omega, epsilon):
        val = np.empty([self.N_dof, self.N_dof], dtype=complex)
        current_row = 0
        for k, (grid_k, space_k, c_k) in enumerate(
            zip(self.grid_list, self.space_list, self.center_list)
        ):
            N_k = grid_k.number_of_elements
            current_col = 0
            for l, (grid_l, space_l, c_l) in enumerate(
                zip(self.grid_list, self.space_list, self.center_list)
            ):
                N_l = grid_l.number_of_elements
                slp_omega_kl = bempp.api.operators.boundary.helmholtz.single_layer(
                    space_k, space_l, space_l, omega, epsilon, c_k, c_l
                )
                rows = slice(current_row, current_row + N_k)
                cols = slice(current_col, current_col + N_l)
                val[rows, cols] = np.transpose(slp_omega_kl.weak_form().A)
                current_col += N_l
            current_row += N_k
        return epsilon**4 * val

    def density(self, epsilon):
        G = self.RHS(epsilon)
        A = lambda s: self.A_Galerkin(s, epsilon)
        val = cq(A, G, self.dt, self.rho, self.N, self.multistep_method)
        return val

    def potential(self, density, epsilon, omega, c):
        space = density.space
        slp_pot_scaled = bempp.api.operators.potential.helmholtz.single_layer_scaled(
            space, self.x_0, omega, epsilon, c
        )
        return epsilon**2 * slp_pot_scaled.evaluate(density)

    def scattered_field(self, epsilon):
        J = self.density(epsilon)
        val = np.zeros((self.N + 1), dtype=complex)
        current_col = 0
        for k, (space_k, c_k) in enumerate(zip(self.space_list, self.center_list)):
            N_k = space_k.grid.number_of_elements
            col = slice(current_col, current_col + N_k)
            J_k = J[:, col]
            sigma_k = lambda G: bempp.api.GridFunction(space_k, coefficients=G)
            A = lambda s, G: self.potential(sigma_k(G), epsilon, s, c_k)
            val += cq_potential_BEM(
                A, J_k, self.dt, self.rho, self.N, self.multistep_method
            )
            current_col += N_k
        return val
