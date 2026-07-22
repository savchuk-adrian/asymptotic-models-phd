import numpy as np
import bempp.api
from numba import jit

from app.utils.geometry import select_geometry, get_grid_points, get_grid_indices
from app.config.incident_fields import define_incident_field


class BemSgflFD:
    def __init__(self, parameters=None):
        self.geometry_large = parameters.geometry_large
        self.geometry_small = parameters.geometry_small
        self.center_large = parameters.center_large
        self.centers = parameters.centers
        self.M = len(self.centers)
        self.omega = parameters.frequency

        self.u_inc = parameters.inc_field
        self.x_0 = parameters.x_0

        self.grid_large = select_geometry(self.geometry_large, self.center_large)
        self.space_large = bempp.api.function_space(self.grid_large[0], "DP", 0)
        self.Ndof = self.space_large.global_dof_count

        self.grid_list = select_geometry(self.geometry_small, self.centers)
        self.sigma_list = self.compute_sigma_list()
        self.capacitances = self.compute_capacitances()

    def compute_sigma_list(self):
        val = np.empty(self.M, dtype=object)
        for k, (grid_k) in enumerate(self.grid_list):
            space_k = bempp.api.function_space(grid_k, "DP", 0)
            slp_0 = bempp.api.operators.boundary.laplace.single_layer(
                space_k, space_k, space_k
            )
            one_grid_fun = bempp.api.GridFunction.from_ones(space_k)
            sigma_k, info = bempp.api.linalg.gmres(slp_0, one_grid_fun, tol=1e-12)
            val[k] = bempp.api.GridFunction(space_k, coefficients=sigma_k.coefficients)
        return val

    def compute_capacitances(self):
        val = np.empty(self.M, dtype=float)
        for k, sigma_k in enumerate(self.sigma_list):
            val[k] = sigma_k.integrate()
        return val

    def compute_potentials(self):
        """We compute integrals of the following function (x_0 - c_k) * (y - c_k) sigma_k(y), * stands for the inner product"""
        val = np.empty(self.M, dtype=object)
        x_0 = self.x_0.reshape(1, 3)
        x_0 = np.array(x_0).flatten()
        for k, (sigma_k, c_k) in enumerate(zip(self.sigma_list, self.center_list)):
            space_k = sigma_k.space

            @bempp.api.complex_callable
            def second_member(x, n, domain_index, result):
                result[0] = (
                    (x_0[0] - c_k[0]) * (x[0] - c_k[0])
                    + (x_0[1] - c_k[1]) * (x[1] - c_k[1])
                    + (x_0[2] - c_k[2]) * (x[2] - c_k[2])
                )

            val[k] = prod_grid_fun(
                bempp.api.GridFunction(space_k, fun=second_member), sigma_k
            )
        return val

    def RHS(self, epsilon):
        u_inc = define_incident_field(**self.u_inc)

        @bempp.api.complex_callable
        def second_member(x, n, domain_index, result):
            result[0] = u_inc(x[0], x[1], x[2])

        g_large = bempp.api.GridFunction(self.space_large, fun=second_member)
        g_large_proj = g_large.projections(self.space_large)

        g_small = np.empty(self.M, dtype=complex)
        for k, (cap_k, c_k) in enumerate(zip(self.capacitances, self.centers)):
            g_small[k] = epsilon * u_inc(c_k[0], c_k[1], c_k[2]) * cap_k

        return np.hstack((g_large_proj, g_small))

    def A_Galerkin(self, epsilon):
        """computing the inverse of the Galerkin Foldy-Lax matrix"""
        val = np.empty((self.M, self.M), dtype=complex)
        for k, (sigma_k, c_k) in enumerate(zip(self.sigma_list, self.centers)):
            for l, (sigma_l, c_l) in enumerate(zip(self.sigma_list, self.centers)):
                space_k = sigma_k.space
                space_l = sigma_l.space
                slp_omega_kl = bempp.api.operators.boundary.helmholtz.single_layer(
                    space_k, space_l, space_l, self.omega, epsilon, c_k, c_l
                )
                slp_omega_kl = slp_omega_kl * sigma_k
                val[k, l] = (
                    slp_omega_kl.projections() @ sigma_l.grid_coefficients
                )  # eps^4 from double integrals and eps^{-2} from the equilibrium densities singma_k
                if k != l:
                    val[l, k] = val[k, l]
        return epsilon**2 * val

    def A_C(self, epsilon):
        """computing the inverse of the coupling matrix"""
        val = np.empty((self.Ndof, self.M), dtype=complex)
        omega = self.omega
        for k, (c_k, cap_k) in enumerate(zip(self.centers, self.capacitances)):

            c_k = np.array(c_k).flatten()

            @bempp.api.complex_callable
            def G_i(x, n, domain_index, result):
                r = np.linalg.norm(x - c_k)
                result[0] = np.exp(1j * omega * r) / (4 * np.pi * r)

            g_func = bempp.api.GridFunction(self.space_large, fun=G_i)

            val[:, k] = epsilon * cap_k * g_func.projections()
        return val

    def density(self, epsilon):
        RHS = self.RHS(epsilon)
        A_BEM = bempp.api.operators.boundary.helmholtz.single_layer(
            self.space_large, self.space_large, self.space_large, self.omega
        )
        A_coupled = self.A_C(epsilon)
        A_G = self.A_Galerkin(epsilon)
        A_BEM_coupled = np.hstack((A_BEM.weak_form().A, A_coupled))
        A_coupled_G = np.hstack((A_coupled.T, A_G))
        A_full = np.vstack((A_BEM_coupled, A_coupled_G))
        val = np.linalg.solve(A_full, RHS)
        return val

    def potential(self, density, epsilon, c):
        space = density.space
        slp_pot_scaled = bempp.api.operators.potential.helmholtz.single_layer_scaled(
            space, self.x_0, self.omega, epsilon, c
        )
        return epsilon * slp_pot_scaled.evaluate(density)

    def scattered_field(self, epsilon):
        val = 0
        J = self.density(epsilon)
        density_large = bempp.api.GridFunction(
            self.space_large, coefficients=J[: self.Ndof]
        )
        slp_pot_large = bempp.api.operators.potential.helmholtz.single_layer_scaled(
            self.space_large, self.x_0, self.omega
        )
        val += slp_pot_large.evaluate(density_large)
        for k, (sigma_k, c_k) in enumerate(zip(self.sigma_list, self.centers)):
            val += self.potential(J[self.Ndof + k] * sigma_k, epsilon, c_k)
        return val

    def scattered_field_on_plane(
        self, epsilon, N_grid, xmin, xmax, ymin, ymax, plane="XY"
    ):
        J = self.density(epsilon)
        points = get_grid_points(N_grid, xmin, xmax, ymin, ymax, plane)
        idx = get_grid_indices(
            points, self.geometry_large, self.center_large[0], 1.0, plane
        )

        val = np.full(points.shape[1], np.nan, dtype=np.complex128)

        for c_k in self.centers:
            idx_k = get_grid_indices(points, self.geometry_small, c_k, epsilon, plane)
            idx = idx & idx_k

        slp_pot_large = bempp.api.operators.potential.helmholtz.single_layer_scaled(
            self.space_large, points[:, idx], self.omega
        )
        sigma = bempp.api.GridFunction(self.space_large, coefficients=J[: self.Ndof])
        res = slp_pot_large.evaluate(sigma)
        for k, (sigma_k, c_k) in enumerate(zip(self.sigma_list, self.centers)):
            J_k = J[self.Ndof + k]
            space_k = sigma_k.space
            slp_k = bempp.api.operators.potential.helmholtz.single_layer_scaled(
                space_k, points[:, idx], self.omega, epsilon, c_k
            )
            res += epsilon * slp_k.evaluate(J_k * sigma_k)

        val[idx] = np.nan_to_num(val[idx], nan=0) + res.flat

        return val.reshape((N_grid, N_grid))


def prod_grid_fun(f, g):
    val = f.projections(g.space) @ g.coefficients
    return val
