import numpy as np
import bempp.api
from app.utils.cq_solver import cq, cq_potential, cq_potential_BEM
from app.utils.geometry import select_geometry
from app.config.incident_fields import define_incident_field

eps_m = 10**(-12)

class CouplingTD:
    def __init__(self, parameters=None):
        # Geometry setup
        self.geometry_large = parameters.geometry_large
        self.geometry_small = parameters.geometry_small
        self.center_large = parameters.center_large
        self.centers = parameters.centers
        self.M = len(self.centers)
        
        # Time-domain setup
        self.dt = parameters.dt
        self.T = parameters.T
        self.multistep_method = parameters.multistep_method
        self.N = int(self.T / self.dt)
        self.t = np.linspace(0, self.T, self.N + 1)
        self.rho = eps_m**(0.5 / (self.N + 1))

        self.u_inc_params = parameters.inc_field
        self.x_0 = parameters.x_0

        # Large particle space
        self.grid_large = select_geometry(self.geometry_large, self.center_large)
        self.space_large = bempp.api.function_space(self.grid_large[0], "DP", 0)
        self.Ndof = self.space_large.global_dof_count

        # Small particles setup
        self.grid_list = select_geometry(self.geometry_small, self.centers)
        self.sigma_list = self.compute_sigma_list()
        self.capacitances = self.compute_capacitances()

    def compute_sigma_list(self):
        val = np.empty(self.M, dtype=object) 
        for k, grid_k in enumerate(self.grid_list):
            space_k = bempp.api.function_space(grid_k, "DP", 0)
            slp_0 = bempp.api.operators.boundary.laplace.single_layer(space_k, space_k, space_k)
            one_grid_fun = bempp.api.GridFunction.from_ones(space_k)
            sigma_k, _ = bempp.api.linalg.gmres(slp_0, one_grid_fun, tol=1e-12)
            val[k] = bempp.api.GridFunction(space_k, coefficients=sigma_k.coefficients)
        return val

    def compute_capacitances(self):
        val = np.empty(self.M, dtype=float)
        for k, sigma_k in enumerate(self.sigma_list):
            val[k] = sigma_k.integrate()
        return val

    def RHS(self, epsilon):
        """Matrix-valued RHS: rows are time-steps, columns are DOFs (Large + Small)"""
        u_inc = define_incident_field(**self.u_inc_params)
        total_dofs = self.Ndof + self.M
        val = np.empty((self.N + 1, total_dofs), dtype=complex)
        c = self.center_large[0]
        for n, t_n in enumerate(self.t):
            # Large particle contribution
            @bempp.api.complex_callable
            def second_member(x, n_vec, domain_index, result):
                result[0] = u_inc(epsilon*(x[0]-c[0])+c[0], epsilon*(x[1]-c[1])+c[1], epsilon*(x[2]-c[2])+c[2], t_n)
            
            g_large = bempp.api.GridFunction(self.space_large, fun=second_member)
            val[n, :self.Ndof] = g_large.projections(self.space_large)

            # Small particles contribution
            for k, (cap_k, c_k) in enumerate(zip(self.capacitances, self.centers)):
                val[n, self.Ndof + k] = epsilon * u_inc(c_k[0], c_k[1], c_k[2], t_n) * cap_k
        return val

    def A_block_operator(self, s, epsilon):
        """Constructs the block matrix for a specific Laplace frequency s"""
        # 1. Large Particle BEM block
        A_BEM = bempp.api.operators.boundary.helmholtz.single_layer(
            self.space_large, self.space_large, self.space_large, s).weak_form().A

        # 2. Coupling block (Large-Small)
        A_C = np.empty((self.Ndof, self.M), dtype=complex)
        for k, (sigma_k, c_k) in enumerate(zip(self.sigma_list, self.centers)):
            space_k = sigma_k.space
            slp_coupled = bempp.api.operators.boundary.helmholtz.single_layer_coupled(
                space_k, self.space_large, self.space_large, s, epsilon, c_k)
            A_C[:, k] = slp_coupled.weak_form().A @ sigma_k.coefficients
        A_C *= epsilon

        # 3. Small Particles Galerkin block (Small-Small)
        A_G = np.empty((self.M, self.M), dtype=complex)   
        for k, sigma_k in enumerate(self.sigma_list):
            c_k = self.centers[k]
            for l, sigma_l in enumerate(self.sigma_list):      
                c_l = self.centers[l]
                slp_kl = bempp.api.operators.boundary.helmholtz.single_layer(
                    sigma_k.space, sigma_l.space, sigma_l.space, s, epsilon, c_k, c_l)   
                slp_kl_fun = slp_kl * sigma_k
                A_G[k, l] = slp_kl_fun.projections() @ sigma_l.grid_coefficients
        A_G *= epsilon**2

        # Assembly
        top_block = np.hstack((A_BEM, A_C))
        bottom_block = np.hstack((A_C.T, A_G))
        return np.vstack((top_block, bottom_block))

    def density(self, epsilon):
        G = self.RHS(epsilon)
        A_op = lambda s: self.A_block_operator(s, epsilon)
        val = cq(A_op, G, self.dt, self.rho, self.N, self.multistep_method)
        return val

    def scattered_field(self, epsilon):
        J_total = self.density(epsilon) # shape (N+1, Ndof + M)
        val = np.zeros(self.N + 1, dtype=complex)

        density_large = lambda G: bempp.api.GridFunction(self.space_large, coefficients=G)
        large_pot_op = lambda s, G: bempp.api.operators.potential.helmholtz.single_layer_scaled(self.space_large, self.x_0, s).evaluate(density_large(G))

        val += cq_potential_BEM(large_pot_op, J_total[:, :self.Ndof], self.dt, self.rho, self.N, self.multistep_method)

        # Potential from Small Particles
        for k, (sigma_k, c_k) in enumerate(zip(self.sigma_list, self.centers)):
            space_k = sigma_k.space

            small_pot_op = lambda s: bempp.api.operators.potential.helmholtz.single_layer_scaled(space_k, self.x_0, s, epsilon, c_k).evaluate(sigma_k)

            val += cq_potential(small_pot_op, J_total[:, self.Ndof + k], self.dt, self.rho, self.N, self.multistep_method)

        return val