import numpy as np
import bempp.api

from utils.cq_solver import cq, cq_potential, cq_potential_plane
from utils.geometry import select_geometry, get_grid
from config.incident_fields import define_incident_field
from config.settings import SimulationConfigTD, SimulationConfigPlaneTD


class GalerkinFoldyLaxSolver:
    """
    This class implements a solver for the Galerkin Foldy-Lax model:

    u^eps_sca(x, t) = sum_{k=1}^N (S_{pot, k} * [lambda^eps_l sigma^eps_k])(x, t),

    where:

    * S_{pot, k} denotes the retarded potential operator restricted to a particle Gamma^eps_k:
        (S_{pot, k} * phi)(x, t) = int_{Gamma^eps_k} \frac{phi(y, t - |x - y|)}{4 pi |x - y|} dGamma_y.

    * sigma^eps_k denotes the equilibrium densities defined on Gamma^eps_k, i.e., the unique solution
        of the single-layer boundary integral equation for the Laplace problem with a constant
        right-hand side:
        int_{Gamma^eps_k} \frac{sigma^eps_k(y)}{4 pi |x - y|} dGamma_y = 1, for x \in Gamma^eps_k.

    * lambda^eps_k are time-domain functions that solve the following linear
        convolutional-in-time system:
        sum_{k=1}^N (K^eps_lk * lambda^eps_k)(t) = q^eps_l(t),

    where:

    (K^eps_lk * lambda^eps_k)(t)
    = int_{Gamma^eps_l} int_{Gamma^eps_k} \frac{lambda^eps_k(t - |x - y|)}{4 pi |x - y|} sigma^eps_k(y) sigma^eps_l(x) dGamma_y dGamma_x.

    q^eps_l(t) = - int_{Gamma^eps_l} u^inc(x, t) sigma^eps_l(x) dGamma_x.

    For more details, please see [1, pp. 4-5].

    References
    ----------
    [1] Kachanovska, M., & Savchuk, A. (2025). Asymptotic models for time-domain scattering by
        small particles of arbitrary shapes. https://arxiv.org/pdf/2511.11103.
    """

    def __init__(self, config: SimulationConfigTD | SimulationConfigPlaneTD):
        """
        Initializes the solver with a given configuration.

        Parameters
        ----------
        config : SimulationConfigTD | SimulationConfigPlaneTD
            Configuration object containing parameters: centers of particles,
            geometry type, incident field, and evaluation points.

        Attributes
        ----------
        M : int
            Number of particles in the system.
        N : int
            Number of time steps.
        rho : float
            Contour radius for Convolution Quadrature.
        t : np.ndarray
            Time mesh from 0 to T with N+1 points.
        grids : list of bempp.api.Grid
            List of grids for each particle. Initialized as None.
        equilibriums : list of bempp.api.GridFunction
            Basis of the asymptotic Galerkin space (equilibrium densities sigma^eps_k).
        """
        self.config = config
        self.M = len(config.centers)
        self.N = int(config.T / config.dt)
        self.rho = (1e-12) ** (0.5 / (self.N + 1))

        self.t = np.linspace(0, self.config.T, self.N + 1)

        self.grids = None
        self.equilibriums = None

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
        self.equilibriums = self._compute_equilibriums()

    def _compute_equilibriums(self):
        """
        Computes the equilibrium densities sigma^eps_k for each particle.

        This is done by solving the single-layer Laplace boundary integral
        equation with a constant right-hand side (equal to 1) for each particle.
        """
        val = np.empty(self.M, dtype=object)
        for k, (grid_k) in enumerate(self.grids):
            space_k = bempp.api.function_space(grid_k, "DP", 0)
            slp_0 = bempp.api.operators.boundary.laplace.single_layer(
                space_k, space_k, space_k
            )
            one_grid_fun = bempp.api.GridFunction.from_ones(space_k)
            sigma_k, _ = bempp.api.linalg.gmres(slp_0, one_grid_fun, tol=1e-12)
            val[k] = bempp.api.GridFunction(space_k, coefficients=sigma_k.coefficients)
        return val

    def rhs(self, epsilon):
        """
        Computes the right-hand side of the convolutional in time system:

        q^eps_k(t) = - int_{Gamma^eps_k} u^inc(x, t) sigma^eps_k(x) dGamma_x.

        Parameters
        ----------
        epsilon : float
            Scaling parameter in the range (0, 1].

        Returns
        -------
        val : np.ndarray
            A complex-valued NumPy array of dimension (N+1, M), where N+1 is the
            number of time steps and M is the number of particles. The elements
            of the matrix val are calculated by projecting the incident field onto
            the equilibrium densities for each particle k at each time step t.
        """
        self._ensure_initialized()
        val = np.empty((self.N + 1, self.M), dtype=complex)
        u_inc = define_incident_field(**self.config.inc_field)
        for k, (sigma_k, c_k) in enumerate(zip(self.equilibriums, self.config.centers)):
            space_k = sigma_k.space
            for n, t_n in enumerate(self.t):

                @bempp.api.complex_callable
                def second_member(x, n, domain_index, result):
                    result[0] = u_inc(
                        epsilon * (x[0] - c_k[0]) + c_k[0],
                        epsilon * (x[1] - c_k[1]) + c_k[1],
                        epsilon * (x[2] - c_k[2]) + c_k[2],
                        t_n,
                    )

                g_dt_k = bempp.api.GridFunction(space_k, fun=second_member)
                val[n, k] = np.dot(
                    g_dt_k.projections(space_k), sigma_k.grid_coefficients
                )
        return epsilon * val

    def galerkin_matrix(self, omega: complex, epsilon: float):
        """
        Constructs the Laplace-domain Galerkin matrix.

        This matrix is the Laplace-domain representation of the time-domain
        convolutional operator K. In the Convolution Quadrature (CQ) scheme,
        this matrix is computed for various complex frequencies (omega) and
        then used to reconstruct the time-domain density lambda^eps.

        The entries A_kl(omega) represent the interaction between particle k
        and particle l at a given Laplace transform parameter omega:

        A_kl(omega) = \int_{\Gamma^\varepsilon_l} (S_kl(omega) \sigma_k)(x) \sigma_l(x) d\Gamma_x

        Parameters
        ----------
        epsilon : float
            Scaling parameter in the range (0, 1].
        omega : complex
            Complex Laplace transform parameter.

        Returns
        -------
        mat : np.ndarray
            A complex-valued NumPy array of dimension (M, M).
        """
        self._ensure_initialized()
        mat = np.empty((self.M, self.M), dtype=complex)

        for k in range(self.M):
            sigma_k = self.equilibriums[k]
            c_k = self.config.centers[k]
            space_k = sigma_k.space

            for l in range(k, self.M):
                sigma_l = self.equilibriums[l]
                c_l = self.config.centers[l]
                space_l = sigma_l.space

                slp = bempp.api.operators.boundary.helmholtz.single_layer(
                    space_k, space_l, space_l, omega, epsilon, c_k, c_l
                )

                res_fun = slp * sigma_k
                val_kl = res_fun.projections() @ sigma_l.grid_coefficients

                mat[k, l] = val_kl
                if k != l:
                    mat[l, k] = val_kl

        return epsilon**2 * mat

    def density(self, epsilon):
        """
        Computes the unknown density coefficients lambda^eps_k(t) by solving
        the convolutional-in-time system.

        This method solves the following system of convolutional equations:
            sum_{k=1}^M (K_lk * lambda^eps_k)(t) = q^eps_l(t),  l = 1,...,M.

        The solution is obtained using the Convolution Quadrature (CQ) method.

        Parameters
        ----------
        epsilon : float
            Scaling parameter in the range (0, 1].

        Returns
        -------
        val : np.ndarray
            A complex-valued NumPy array of dimension (N+1, M), where N+1 is the
            number of time steps and M is the number of particles.
        """
        G = self.rhs(epsilon)
        A = lambda s: self.galerkin_matrix(s, epsilon)
        val = cq(A, G, self.config.dt, self.rho, self.N, self.config.multistep_method)
        return val

    def potential(self, density, epsilon, omega, c):
        """
        Evaluates the 3D Helmholtz single-layer potential.

        This method applies the Helmholtz single-layer potential operator to the
        provided density and evaluates the result at the observation point x_0
        defined in the configuration.

        Parameters
        ----------
        density : bempp.api.GridFunction
            The density defined on the particle surface.
        epsilon : float
            Scaling parameter.
        omega : complex
            Complex frequency.
        c : tuple
            Center of the particle.

        Returns
        -------
        val : np.ndarray
            The complex-valued potential evaluated at point x_0.
        """
        space = density.space
        slp_pot_scaled = bempp.api.operators.potential.helmholtz.single_layer_scaled(
            space, self.config.x_0, omega, epsilon, c
        )
        return epsilon * slp_pot_scaled.evaluate(density)

    def compute_scattered_field(self, epsilon):
        """
        Computes the approximated scattered field u^eps_app(x_0, t) at the evaluation point.

        This method reconstructs the time-domain scattered field by summing the
        contributions from all M particles:

        u^eps_app(x_0, t) = sum_{k=1}^M (S_{pot, k} * [lambda^eps_{G,k} sigma^eps_k])(x_0, t).

        The convolution between the single-layer potential operator and the time-dependent
        density lambda^eps_k is calculated using the Convolution Quadrature (CQ) method

        Parameters
        ----------
        epsilon : float
            Scaling parameter in the range (0, 1].

        Returns
        -------
        val : np.ndarray
            A complex-valued 1D NumPy array of dimension (N+1,) representing
            the scattered field at the observation point x_0 for each time step.
        """
        J = self.density(epsilon)  # (N+1, M)
        val = np.zeros((self.N + 1), dtype=complex)

        for k, (sigma_k, c_k) in enumerate(zip(self.equilibriums, self.config.centers)):
            A = lambda s: self.potential(sigma_k, epsilon, s, c_k)
            val += cq_potential(
                A,
                J[:, k],
                self.config.dt,
                self.rho,
                self.N,
                self.config.multistep_method,
            )
        return val
