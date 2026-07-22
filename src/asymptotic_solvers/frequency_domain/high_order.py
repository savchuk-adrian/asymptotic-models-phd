from re import L
import numpy as np
import bempp.api

from app.utils.geometry import select_geometry, get_grid
from app.config.incident_fields import define_incident_field


class HighOrderFD:
    def __init__(self, parameters=None):
        self.geometry = parameters.geometry
        self.center_list = parameters.centers
        self.M = len(self.center_list)
        self.omega = parameters.frequency
        self.order = parameters.order

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

        self.space_list = self.compute_space_list()

        self.basis_functions = self.compute_basis_functions()
        self.orthogonal_basis_functions = self.QR()

    def compute_space_list(self):
        val = np.empty(self.M, dtype=object)
        for k, (grid_k) in enumerate(self.grid_list):
            space_k = bempp.api.function_space(grid_k, "DP", 0)
            val[k] = space_k
        return val

    # Creates a list of basis functions for the high order method on one particle
    def compute_basis_functions(self):
        space_keys = []
        N = self.order
        for n in range(self.M):
            space_keys.append("space_" + str(n + 1))
        val = {key: [] for key in space_keys}
        for key, space_j, c_j in zip(val.keys(), self.space_list, self.center_list):
            slp_0 = bempp.api.operators.boundary.laplace.single_layer(
                space_j, space_j, space_j
            )
            for m in range(N + 1):
                for l in range(m + 1):
                    for k in range(l, m + 1):

                        @bempp.api.real_callable
                        def rhs(x, n, domain_index, result):
                            # arg = [(x[0]-c_j[0]), (x[1]-c_j[1]), (x[2]-c_j[2])]
                            # result[0] = arg[0]**(m-k)*arg[1]**(k-l)*arg[2]**l
                            result[0] = (
                                (x[0] - c_j[0]) ** (m - k)
                                * (x[1] - c_j[1]) ** (k - l)
                                * (x[2] - c_j[2]) ** l
                            )

                        rhs_grid_fun = bempp.api.GridFunction(space_j, fun=rhs)
                        sigma_j, info = bempp.api.linalg.gmres(
                            slp_0, rhs_grid_fun, tol=1e-12
                        )
                        # val[key].append(sigma_j)
                        val[key].append((sigma_j, m - 1))
            # part with the 'surprise' term
            if N == 2:
                slp_0 = bempp.api.operators.boundary.laplace.single_layer(
                    space_j, space_j, space_j
                )
                one_grid_fun = bempp.api.GridFunction.from_ones(space_j)
                sigma, info = bempp.api.linalg.gmres(slp_0, one_grid_fun, tol=1e-12)
                slp_custom = bempp.api.operators.boundary.helmholtz.custom(
                    space_j, space_j, space_j, N
                )
                phi = slp_custom * sigma
                surprise_term, info = bempp.api.linalg.gmres(slp_0, phi, tol=1e-12)
                val[key].append((surprise_term, 2))
        return val

    def QR(self):
        val = {}
        for (key, sigma_list_k), space_k in zip(
            self.basis_functions.items(), self.space_list
        ):
            if not sigma_list_k:
                val[key] = []
                continue
            groups = {}
            for sigma_j, m in sigma_list_k:
                if m not in groups:
                    groups[m] = []
                groups[m].append(sigma_j)
            new_list = []
            for m, group in groups.items():
                columns = [sigma.coefficients for sigma in group]
                A = np.column_stack(columns)
                Q, _ = np.linalg.qr(A)
                for i in range(Q.shape[1]):
                    sigma_qr = bempp.api.GridFunction(space_k, coefficients=Q[:, i])
                    new_list.append((sigma_qr, m))
            val[key] = new_list
        return val

    def slp_matrix(self, epsilon):
        val = np.empty((self.M, self.M), dtype=object)
        for k, (space_k, c_k) in enumerate(zip(self.space_list, self.center_list)):
            for l, (space_l, c_l) in enumerate(zip(self.space_list, self.center_list)):
                val[k, l] = bempp.api.operators.boundary.helmholtz.single_layer(
                    space_k, space_l, space_l, self.omega, epsilon, c_k, c_l
                )
                if k != l:
                    val[l, k] = bempp.api.operators.boundary.helmholtz.single_layer(
                        space_l, space_k, space_k, self.omega, epsilon, c_l, c_k
                    )  # val[k,l]
        return val

    def A_Galerkin(self, epsilon):
        N = sum(len(items) for items in self.orthogonal_basis_functions.values())
        N_base = len(self.orthogonal_basis_functions["space_1"])
        val = np.empty((N, N), dtype=complex)
        _slp_matrix = self.slp_matrix(epsilon)
        for k, pair_list_k in enumerate(self.orthogonal_basis_functions.values()):
            for i, (sigma_i, m1) in enumerate(pair_list_k):
                for l, pair_list_l in enumerate(
                    self.orthogonal_basis_functions.values()
                ):
                    for j, (sigma_j, m2) in enumerate(pair_list_l):
                        slp_omega_kl = _slp_matrix[k, l] * sigma_i
                        val[N_base * k + i, N_base * l + j] = (
                            epsilon ** (m1 + m2 + 4)
                            * slp_omega_kl.projections()
                            @ sigma_j.grid_coefficients
                        )
                        if i != j:
                            val[N_base * l + j, N_base * k + i] = val[
                                N_base * k + i, N_base * l + j
                            ]
        return val

    def RHS(self, epsilon):
        N_base = len(self.orthogonal_basis_functions["space_1"])
        N = sum(len(items) for items in self.orthogonal_basis_functions.values())
        u_inc = define_incident_field(**self.u_inc)
        val = np.empty(N, dtype=complex)
        for k, (pairs_k, c_k) in enumerate(
            zip(self.orthogonal_basis_functions.values(), self.center_list)
        ):
            for i, (sigma_i, power) in enumerate(pairs_k):
                space_k = sigma_i.space

                @bempp.api.complex_callable
                def second_member(x, n, domain_index, result):
                    result[0] = u_inc(
                        epsilon * (x[0] - c_k[0]) + c_k[0],
                        epsilon * (x[1] - c_k[1]) + c_k[1],
                        epsilon * (x[2] - c_k[2]) + c_k[2],
                    )

                g_k = bempp.api.GridFunction(space_k, fun=second_member)
                val[N_base * k + i] = (
                    epsilon ** (power + 2)
                    * g_k.projections(space_k)
                    @ sigma_i.grid_coefficients
                )
        return val

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
        return slp_pot_scaled.evaluate(density)

    def scattered_field(self, epsilon):
        N_base = len(self.orthogonal_basis_functions["space_1"])
        J = self.density(epsilon)
        val = 0
        for k, (pairs_k, c_k) in enumerate(
            zip(self.orthogonal_basis_functions.values(), self.center_list)
        ):
            for i, (sigma_i, power) in enumerate(pairs_k):
                val += epsilon ** (power + 2) * self.potential(
                    J[N_base * k + i] * sigma_i, epsilon, c_k
                )
        return val

    def scattered_field_on_plane(
        self, epsilon, N_grid, xmin, xmax, ymin, ymax, plane="XY"
    ):
        J = self.density(epsilon)
        points, idx = get_grid(
            self.geometry,
            self.center_list,
            N_grid,
            xmin,
            xmax,
            ymin,
            ymax,
            epsilon=epsilon,
            plane=plane,
        )
        val = np.full(points.shape[1], np.nan, dtype=np.complex128)
        N_base = len(self.orthogonal_basis_functions["space_1"])

        for k, (pairs_k, space_k, c_k) in enumerate(
            zip(
                self.orthogonal_basis_functions.values(),
                self.space_list,
                self.center_list,
            )
        ):
            slp_k = bempp.api.operators.potential.helmholtz.single_layer_scaled(
                space_k, points[:, idx], self.omega, epsilon, c_k
            )
            for i, (sigma_i, power) in enumerate(pairs_k):
                res = (
                    epsilon ** (power + 2) * J[N_base * k + i] * slp_k.evaluate(sigma_i)
                )

                val[idx] = np.nan_to_num(val[idx], nan=0) + res.flat

        return val.reshape((N_grid, N_grid))
