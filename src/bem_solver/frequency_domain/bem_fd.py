import numpy as np
import bempp.api

from bempp.api.assembly.blocked_operator import BlockedOperator

from app.utils.geometry import select_geometry, get_grid
from app.config.incident_fields import define_incident_field

class BemFD:
    def __init__(self, parameters=None):
        self.geometry = parameters.geometry
        self.center_list = parameters.centers
        self.M = len(self.center_list)
        self.omega = parameters.frequency
        
        self.u_inc = parameters.inc_field
        self.x_0 = parameters.x_0

        if parameters.directions is not None:
            self.grid_list = select_geometry(self.geometry, self.center_list, directions=parameters.directions, angles=parameters.angles)
        # elif parameters.h is not None:
        #     self.grid_list = select_geometry(self.geometry, self.center_list, h=parameters.h)
        else:
            self.grid_list = select_geometry(self.geometry, self.center_list)
        
        self.space_list = self.compute_space_list()

    def compute_space_list(self):
        val = np.empty(self.M, dtype=object) 
        for k, (grid_k) in enumerate(self.grid_list):
            space_k = bempp.api.function_space(grid_k, "DP", 0)
            val[k] = space_k
        return val

    def RHS(self, epsilon):
        u_inc = define_incident_field(**self.u_inc)
        val = []
        for k, (space_k, c_k) in enumerate(zip(self.space_list, self.center_list)):  
            @bempp.api.complex_callable
            def second_member(x, n, domain_index, result):
                result[0] = u_inc(epsilon*(x[0]-c_k[0])+c_k[0], epsilon*(x[1]-c_k[1])+c_k[1], epsilon*(x[2]-c_k[2])+c_k[2])
            g_k = bempp.api.GridFunction(space_k, fun=second_member)
            val.append(g_k) 
        return val
    
    def A_Galerkin(self, epsilon):
        val = BlockedOperator(self.M, self.M)
        for k, (space_k, c_k) in enumerate(zip(self.space_list, self.center_list)):
            for l, (space_l, c_l) in enumerate(zip(self.space_list, self.center_list)):
                slp_omega_kl = bempp.api.operators.boundary.helmholtz.single_layer(space_k, space_l, space_l, self.omega, epsilon, c_k, c_l)  
                val[l, k] = slp_omega_kl
        return val

    def density(self, epsilon):
        G = self.RHS(epsilon)
        A = self.A_Galerkin(epsilon)
        val = bempp.api.linalg.gmres(A, G, tol=1e-5)
        return val[0]
      
    def potential(self, density, epsilon, c):
        space = density.space 
        slp_pot_scaled = bempp.api.operators.potential.helmholtz.single_layer_scaled(space, self.x_0, self.omega, epsilon, c)
        return slp_pot_scaled.evaluate(density)

    def scattered_field(self, epsilon):
        J = self.density(epsilon)
        val = 0
        for k, c_k in enumerate(self.center_list):
            val += self.potential(J[k], epsilon, c_k)
        return val 
    
    def scattered_field_on_plane(self, epsilon, N_grid, xmin, xmax, ymin, ymax, plane='XY'):
        J = self.density(epsilon)
        points, idx = get_grid(self.geometry, self.center_list, N_grid, xmin, xmax, ymin, ymax, epsilon = epsilon, plane=plane)
        val = np.full(points.shape[1], np.nan, dtype=np.complex128)

        for k, (space_k, c_k) in enumerate(zip(self.space_list, self.center_list)):
            slp_k = bempp.api.operators.potential.helmholtz.single_layer_scaled(space_k, points[:, idx], self.omega, epsilon, c_k)
            res = slp_k.evaluate(J[k])
            val[idx] = np.nan_to_num(val[idx], nan=0) + res.flat
        return val.reshape((N_grid, N_grid))
