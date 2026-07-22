import numpy as np
import bempp.api

from app.utils.geometry import select_geometry, get_grid_points, get_grid_indices
from app.config.incident_fields import define_incident_field
from old_code.modul_frequency_domain import sigma_high_order


class BemCoupledFD:
    def __init__(self, parameters = None):
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
        self.NdofL = self.grid_large[0].number_of_elements

        self.grid_list = select_geometry(self.geometry_small, self.centers)
        self.N_dof_small = sum(grid_k.number_of_elements for grid_k in self.grid_list)
        self.spaces_small = self.compute_space_list()

    def compute_space_list(self):
        val = np.empty(self.M, dtype=object) 
        for k, (grid_k) in enumerate(self.grid_list):
            space_k = bempp.api.function_space(grid_k, "DP", 0)
            val[k] = space_k
        return val

    def RHS(self, epsilon):
        u_inc = define_incident_field(**self.u_inc)

        val = np.empty([0, 1], dtype=complex)

        @bempp.api.complex_callable
        def second_member(x, n, domain_index, result):
            result[0] = u_inc(x[0], x[1], x[2])
        g_L = bempp.api.GridFunction(self.space_large, fun=second_member)
        val = np.vstack((val, g_L.projections(self.space_large).reshape(-1, 1)))

        for space_k, c_k in zip(self.spaces_small, self.centers): 
            @bempp.api.complex_callable
            def second_member(x, n, domain_index, result):
                result[0] = u_inc(epsilon*(x[0]-c_k[0])+c_k[0], epsilon*(x[1]-c_k[1])+c_k[1], epsilon*(x[2]-c_k[2])+c_k[2])
            g_k = bempp.api.GridFunction(space_k, fun=second_member)
            val_k = epsilon ** 2 * g_k.projections(space_k).reshape(-1, 1)
            val = np.vstack((val, val_k))
        return val
    
    def A_scaled(self, epsilon):
        val = np.empty([0, self.N_dof_small], dtype=complex)
        for space_k, c_k in zip(self.spaces_small, self.centers):
            N_k = space_k.global_dof_count
            val_k = np.empty([N_k, 0], dtype=complex)
            for space_l, c_l in zip(self.spaces_small, self.centers):
                slp_omega_kl = bempp.api.operators.boundary.helmholtz.single_layer(
                    space_l, space_k, space_k, self.omega, epsilon, c_l, c_k
                )
                A_kl = slp_omega_kl.weak_form().A
                #print("A_kl: ", A_kl.shape, "val_k: ", val_k.shape)
                val_k = np.hstack((val_k, A_kl))  
            val = np.vstack((val, val_k))

        return epsilon ** 4 * val
    
    def A_large(self):
        slp_omega_large = bempp.api.operators.boundary.helmholtz.single_layer(self.space_large, self.space_large, self.space_large, self.omega)  
        return slp_omega_large.weak_form().A
    
    def A_coupled(self, epsilon):
        val = np.empty([self.NdofL, 0], dtype=complex)
        for (space_k, c_k) in zip(self.spaces_small, self.centers):
            slp_omega_k = bempp.api.operators.boundary.helmholtz.single_layer_coupled(space_k, self.space_large, self.space_large, self.omega, epsilon, c_k) 
            val = np.hstack((val, slp_omega_k.weak_form().A))
        return epsilon ** 2 * val
    
    def A_total(self, epsilon):
        A_coupled = self.A_coupled(epsilon)
        A_scaled = self.A_scaled(epsilon)
        A_large = self.A_large()
        block_1 = np.hstack((A_large, A_coupled))
        block_2 = np.hstack((A_coupled.T, A_scaled))
        val = np.vstack((block_1, block_2))
        return val

    def density(self, epsilon):
        G = self.RHS(epsilon)
        A = self.A_total(epsilon)
        val = np.linalg.solve(A, G)
        return val.flatten() 
      
    def potential(self, density, epsilon, c):
        space = density.space 
        slp_pot_scaled = bempp.api.operators.potential.helmholtz.single_layer_scaled(space, self.x_0, self.omega, epsilon, c)
        return epsilon ** 2 * slp_pot_scaled.evaluate(density)

    def scattered_field(self, epsilon):
        J = self.density(epsilon)
        val = 0

        slp_pot_large = bempp.api.operators.potential.helmholtz.single_layer_scaled(self.space_large, self.x_0, self.omega)
        phi_large = bempp.api.GridFunction(self.space_large, coefficients = J[:self.NdofL])
        val = slp_pot_large.evaluate(phi_large)
        current_col = self.NdofL
        for (space_k, c_k) in zip(self.spaces_small, self.centers):
            N_k = space_k.grid.number_of_elements
            col = slice(current_col, current_col + N_k) 
            J_k = J[col]
            sigma_k = bempp.api.GridFunction(space_k, coefficients = J_k)
            val += self.potential(sigma_k, epsilon, c_k)
            current_col += N_k
        return val 
    
    def scattered_field_on_plane(self, epsilon, N_grid, xmin, xmax, ymin, ymax, plane='XY'):
        J = self.density(epsilon)
        points = get_grid_points(N_grid, xmin, xmax, ymin, ymax, plane)
        idx = get_grid_indices(points, self.geometry_large, self.center_large[0], 1.0, plane)

        val = np.full(points.shape[1], np.nan, dtype=np.complex128)

        for c_k in self.centers:
            idx_k = get_grid_indices(points, self.geometry_small, c_k, epsilon, plane)
            idx = idx & idx_k
        
        slp_pot_large = bempp.api.operators.potential.helmholtz.single_layer_scaled(self.space_large, points[:, idx], self.omega)
        sigma = bempp.api.GridFunction(self.space_large, coefficients = J[:self.NdofL])
        res = slp_pot_large.evaluate(sigma)
        current_col = self.NdofL
        for k, (space_k, c_k) in enumerate(zip(self.spaces_small, self.centers)):
            N_k = space_k.grid.number_of_elements
            col = slice(current_col, current_col + N_k) 
            J_k = J[col]
            sigma_k = bempp.api.GridFunction(space_k, coefficients = J_k)
            slp_k = bempp.api.operators.potential.helmholtz.single_layer_scaled(space_k, points[:, idx], self.omega, epsilon, c_k)
            res += epsilon ** 2 * slp_k.evaluate(sigma_k)
            current_col += N_k

        val[idx] = np.nan_to_num(val[idx], nan=0) + res.flat

        return val.reshape((N_grid, N_grid))