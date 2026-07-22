import numpy as np
import bempp.api
from numba import jit

from utils.geometry import select_geometry
from app.config.incident_fields import define_incident_field


class GalerkinFoldyLaxFD:
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

    def compute_sigma_list(self):
        val = np.empty(self.M, dtype=object)
        for k, (grid_k) in enumerate(self.grid_list):
            space_k = bempp.api.function_space(grid_k, "DP", 0)
            # slp_0 = bempp.api.operators.boundary.laplace.single_layer(space_k, space_k, space_k)
            # @bempp.api.real_callable
            # def one_function(x, n, domain_index, result):
            #     result[0] = 1.0
            # one_grid_fun = bempp.api.GridFunction(space_k, fun=one_function)
            # sigma_k, info = bempp.api.linalg.gmres(slp_0, one_grid_fun, tol=1e-12)
            val[k] = bempp.api.GridFunction.from_ones(space_k)
            # val[k] = bempp.api.GridFunction(space_k, coefficients=sigma_k.coefficients)
        return val

    def compute_capacitance_list(self):
        val = np.empty(self.M, dtype=float)
        for k, sigma_k in enumerate(self.sigma_list):
            val[k] = sigma_k.integrate()
        return val

    def get_grid_for_sphere(self):
        radii = self.compute_capacitance_list() / (4 * np.pi)
        for k, c_k in enumerate(self.center_list):
            grid_k = bempp.api.shapes.sphere(radii[k], origin=c_k, h=0.1)
            self.grid_list[k] = grid_k
        self.sigma_list = self.compute_sigma_list()

    def RHS(self, epsilon):
        val = np.empty(self.M, dtype=complex)
        u_inc = define_incident_field(**self.u_inc)
        for k, (sigma_k, c_k) in enumerate(zip(self.sigma_list, self.center_list)):
            space_k = sigma_k.space

            @bempp.api.complex_callable
            def second_member(x, n, domain_index, result):
                result[0] = u_inc(
                    epsilon * (x[0] - c_k[0]) + c_k[0],
                    epsilon * (x[1] - c_k[1]) + c_k[1],
                    epsilon * (x[2] - c_k[2]) + c_k[2],
                )

            g_k = bempp.api.GridFunction(space_k, fun=second_member)
            val[k] = np.dot(g_k.projections(space_k), sigma_k.grid_coefficients)
        return epsilon * val

    def A_Galerkin(self, epsilon):
        """computing the inverse of the Galerkin Foldy-Lax matrix"""
        val = np.empty((self.M, self.M), dtype=complex)
        for k, (sigma_k, c_k) in enumerate(zip(self.sigma_list, self.center_list)):
            for l, (sigma_l, c_l) in enumerate(zip(self.sigma_list, self.center_list)):
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

    def density(self, epsilon):
        G = self.RHS(epsilon)
        A = self.A_Galerkin(epsilon)
        val = np.linalg.solve(A, G)
        return val

    def potential(self, density, epsilon, c):
        space = density.space
        slp_pot_scaled = bempp.api.operators.potential.helmholtz.single_layer_scaled(
            space, self.x_0, self.omega, epsilon, c
        )
        return epsilon * slp_pot_scaled.evaluate(density)

    def scattered_field(self, epsilon):
        J = self.density(epsilon)  # (N+1, M)
        val = 0
        for k, (sigma_k, c_k) in enumerate(zip(self.sigma_list, self.center_list)):
            val += self.potential(J[k] * sigma_k, epsilon, c_k)
        return val
