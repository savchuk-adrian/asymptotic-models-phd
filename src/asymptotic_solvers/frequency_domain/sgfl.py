import numpy as np
import bempp.api

from scipy.special import jve, hankel1e

from utils.geometry import select_geometry
from config.incident_fields import define_incident_field, define_incident_field_grad


class EfficientGalerkinFoldyLaxFD:
    def __init__(self, parameters=None):
        self.geometry = parameters.geometry
        self.center_list = parameters.centers
        self.M = len(self.center_list)
        self.omega = parameters.frequency

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

        self.sigma_list = self.compute_sigma_list()
        self.capacitances = self.compute_capacitances()
        self.momentums = self.compute_momentums()
        self.momentums_inc_field = self.compute_momentums_inc_field()

        self.space_list = self.compute_space_list()

        self.N_dof = sum(grid.number_of_elements for grid in self.grid_list)

    def compute_space_list(self):
        val = np.empty(self.M, dtype=object)
        for k, (grid_k) in enumerate(self.grid_list):
            space_k = bempp.api.function_space(grid_k, "DP", 0)
            val[k] = space_k
        return val

    def compute_sigma_list(self):
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

    def compute_capacitances(self):
        val = np.empty(self.M, dtype=float)
        for k, sigma_k in enumerate(self.sigma_list):
            val[k] = sigma_k.integrate()
        return val

    def compute_momentums(self):
        """We compute integrals of the following function (x_0 - c_k) * (y - c_k) sigma_k(y), * stands for the inner product"""
        val = np.empty(self.M, dtype=object)
        x_0 = self.x_0.reshape(1, 3)
        x_0 = np.array(x_0).flatten()
        for k, (sigma_k, c_k) in enumerate(zip(self.sigma_list, self.center_list)):
            space_k = sigma_k.space

            @bempp.api.complex_callable
            def second_member(x, n, domain_index, result):
                result[0] = (
                    (c_k[0] - x_0[0]) * (x[0] - c_k[0])
                    + (c_k[1] - x_0[1]) * (x[1] - c_k[1])
                    + (c_k[2] - x_0[2]) * (x[2] - c_k[2])
                )

            val[k] = prod_grid_fun(
                bempp.api.GridFunction(space_k, fun=second_member), sigma_k
            )
        return val

    def compute_momentums_inc_field(self):
        """We compute integrals of the following function d * (y - c_k) sigma_k(y), * stands for the inner product"""
        val = np.empty(self.M, dtype=object)
        d = self.u_inc["d"]
        for k, (sigma_k, c_k) in enumerate(zip(self.sigma_list, self.center_list)):
            space_k = sigma_k.space

            @bempp.api.complex_callable
            def second_member(x, n, domain_index, result):
                result[0] = (
                    d[0] * (x[0] - c_k[0])
                    + d[1] * (x[1] - c_k[1])
                    + d[2] * (x[2] - c_k[2])
                )

            val[k] = prod_grid_fun(
                bempp.api.GridFunction(space_k, fun=second_member), sigma_k
            )
        return val

    def RHS(self, epsilon):
        val = np.empty(self.M, dtype=complex)
        u_inc = define_incident_field(**self.u_inc)
        grad_u_inc = define_incident_field_grad(**self.u_inc)
        for k, (cap_k, pot_k, c_k) in enumerate(
            zip(self.capacitances, self.momentums_inc_field, self.center_list)
        ):
            val[k] = (
                epsilon * u_inc(c_k[0], c_k[1], c_k[2]) * cap_k
                + epsilon**2 * grad_u_inc(c_k[0], c_k[1], c_k[2]) * pot_k
            )
        return val

    def A_Galerkin(self, epsilon):
        """computing the inverse of the Galerkin Foldy-Lax matrix"""
        radiis = epsilon * self.capacitances / (4 * np.pi)
        val = np.empty((self.M, self.M), dtype=complex)
        for k, (r_k, c_k) in enumerate(zip(radiis, self.center_list)):
            for l, (r_l, c_l) in enumerate(zip(radiis, self.center_list)):
                val[k, l] = (
                    1j
                    * 4
                    * np.pi
                    * r_k**2
                    * self.omega
                    * j0(self.omega * r_k)
                    * h0(self.omega * r_k)
                )
                if k != l:
                    dist = np.linalg.norm(np.array(c_k) - np.array(c_l))
                    val[k, l] = (
                        1j
                        * 4
                        * np.pi
                        * self.omega
                        * r_k
                        * r_l
                        * j0(self.omega * r_k)
                        * j0(self.omega * r_l)
                        * h0(self.omega * dist)
                    )
        return val

    def density(self, epsilon):
        G = self.RHS(epsilon)
        A = self.A_Galerkin(epsilon)
        val = np.linalg.solve(A, G)
        return val

    def get_vectorized_density(self, epsilon):
        density = self.density(epsilon)
        val = np.empty(self.N_dof, dtype=complex)
        current_col = 0
        for k, (sigma_k) in enumerate(self.sigma_list):
            N_k = sigma_k.space.grid.number_of_elements
            col = slice(current_col, current_col + N_k)
            val[col] = density[k] * sigma_k.coefficients
            current_col += N_k
        return epsilon ** (-1) * val

    def scattered_field(self, epsilon):
        J = self.density(epsilon)
        val = 0
        x_0 = self.x_0.reshape(1, 3)
        for k, (c_k, cap_k, pot_k) in enumerate(
            zip(self.center_list, self.capacitances, self.momentums)
        ):
            c_k = np.array(c_k)
            dist_k = np.linalg.norm(x_0 - c_k)
            val += (
                J[k]
                * self.G(dist_k)
                * (
                    epsilon * cap_k
                    + epsilon**2 * (1j * self.omega / dist_k - 1 / dist_k**2) * pot_k
                )
            )
        return val

    def G(self, r):
        return 1j * self.omega * h0(self.omega * r) / (4 * np.pi)

    # def norm_l2(self, epsilon, fun):
    #     norm = 0
    #     current_col = 0
    #     for (space_k, grid_k) in zip(self.space_list, self.grid_list):
    #         N_k = grid_k.number_of_elements
    #         col = slice(current_col, current_col + N_k)
    #         fun_k = fun[col]
    #         mass = space_k.mass_matrix()
    #         norm += np.sqrt(np.abs(fun_k.conjugate().T.dot(mass.dot(fun_k))))
    #         current_col += N_k
    #     return epsilon * norm

    # def norm_h_minus_half(self, epsilon, fun):
    #     norm = 0
    #     current_col = 0
    #     for (space_k, grid_k) in zip(self.space_list, self.grid_list):
    #         N_k = grid_k.number_of_elements
    #         col = slice(current_col, current_col + N_k)
    #         fun_k = fun[col]
    #         slp_0 = bempp.api.operators.boundary.laplace.single_layer(space_k, space_k, space_k).weak_form().A
    #         norm += np.sqrt(np.abs(fun_k.conjugate().T.dot(slp_0.dot(fun_k))))
    #     return epsilon**(3/2) * norm

    # def projection_Q_sigma(self, fun):
    #     current_col = 0
    #     val = np.empty(self.N_dof, dtype=complex)
    #     for k, (sigma_k, grid_k, cap_k) in enumerate(zip(self.sigma_list, self.grid_list, self.capacitances)):
    #         N_k = grid_k.number_of_elements
    #         col = slice(current_col, current_col + N_k)
    #         fun_k = bempp.api.GridFunction(sigma_k.space, coefficients=fun[col])
    #         val[col] = fun_k.integrate()/cap_k*sigma_k.coefficients
    #         current_col += N_k
    #     return val

    # def projection_Q_perp(self, fun):
    #     Q_sigma_fun = self.projection_Q_sigma(fun)
    #     return np.ones(self.Ndof, dtype=complex) - Q_sigma_fun


def j0(z):
    """Exponentially scaled Bessel function"""
    val = np.sqrt(np.pi) / np.sqrt(2 * z) * jve(0.5, z)
    return np.exp(abs(np.imag(z))) * val


def h0(z):
    """Exponentially scaled Hankel function"""
    val = np.sqrt(np.pi) / np.sqrt(2 * z) * hankel1e(0.5, z)
    return np.exp(1j * z) * val


def prod_grid_fun(f, g):
    val = f.projections(g.space) @ g.coefficients
    return val
