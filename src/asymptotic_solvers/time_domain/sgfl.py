import numpy as np
import bempp.api

from utils.cq_solver import cq, cq_potential, cq_potential_plane
from utils.geometry import select_geometry, get_grid
from utils.special_functions import helmholtz_greens_function_3d, j0, h0
from config.incident_fields import define_incident_field, define_incident_field_grad
from config.settings import SimulationConfigTD, SimulationConfigPlaneTD


class SimplifiedGalerkinFoldyLaxSolver:
    """
    This class implements a simplified Galerkin Foldy-Lax (SGFL) model.

    Unlike the standard Galerkin Foldy-Lax model, this approach avoids heavy surface
    computations by using capacitance and first moments of the
    equilibrium densities:

    u^eps_app(x, t) = sum_{k=1}^N lambda^eps_k(t-|x-c_k|)/ (4 pi |x-c_k|) (c^eps_k + (x-c_k) . p^eps_k/|x-c_k|^2)
                    + sum_{k=1}^N partial_t lambda^eps_k(t-|x-c_k|) / (4 pi |x-c_k|^2) (x-c_k) . p^eps_k,

    where:

    * c^eps_k and p^eps_k denote capacitance and dipole moments of the equilibrium densities:

        c^eps_k = int_{Gamma^eps_k} sigma^eps_k(y) dGamma_y,
        p^eps_k = int_{Gamma^eps_k} (y - c_k) sigma^eps_k(y) dGamma_y.

    * lambda^eps_k are time-domain functions that solve the following linear
        convolutional-in-time system:

        sum_{k=1}^N (K^eps_lk * lambda^eps_k)(t) = q^eps_l(t),

    where:

    * q^eps_l(t) = -u^inc(c_l, t) c^eps_l - grad u^inc(c_l, t) . p^eps_l.

    * K^eps_lk(t) is defined via its Fourier-Laplace representation M^eps_lk(omega):

        M^eps_ll(omega) = 4 pi i omega (rho_k)^2 j_0(omega rho_k) h_0(omega rho_k),
        M^eps_lk(omega) = 4 pi i omega rho_l rho_k j_0(omega rho_l) j_0(omega rho_k) h_0(omega |c_l - c_k|), l != k.

    with rho_k = c^eps_k / (4 pi), and j_0, h_0 being spherical Bessel and Hankel functions.

    For more details, please see [1, pp. 6-7].

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
        capacitances : list of float
            List of capacitances c^eps_k of the equilibrium densities sigma^eps_k.
        first_moments : lisf of np.ndarray
            List of first moments p^eps_k of the equilibrium densities sigma^eps_k.
        """
        self.config = config
        self.M = len(config.centers)
        self.N = int(config.T / config.dt)
        self.rho = (1e-12) ** (0.5 / (self.N + 1))

        self.t = np.linspace(0, self.config.T, self.N + 1)

        self.grids = None
        self.equilibriums = None
        self.capacitances = None
        self.first_moments = None

        self.first_moments_inc_field = None

    def _ensure_initialized(self):
        """Internal helper to make sure geometry is ready before calculations."""
        if self.capacitances is None:
            self.initialize_geometry()

        if self.first_moments_inc_field is None:
            self.first_moments_inc_field = self._compute_moments_inc_field()

    def initialize_geometry(self):
        """Precomputes grids and capacitances."""
        self.grids = select_geometry(
            self.config.geometry,
            self.config.centers,
            directions=getattr(self.config, "directions", None),
            angles=getattr(self.config, "angles", None),
        )
        self.equilibriums = self._compute_equilibriums()
        self.capacitances = self._compute_capacitances()
        self.first_moments = self._compute_first_moments()

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

    def _compute_capacitances(self):
        """
        Computes the capacitances c^eps_k of the equilibrium densities sigma^eps_k:

        c^eps_k = int_{Gamma^eps_k} sigma^eps_k(y) dGamma_y.

        Returns
        -------
        val : np.ndarray
            A 1D NumPy array of dimension (M,), where M is the number of particles,
            containing the capacitance for each particle k.
        """
        val = np.empty(self.M, dtype=float)
        for k, sigma_k in enumerate(self.equilibriums):
            val[k] = sigma_k.integrate()
        return val

    def _compute_first_moments(self):
        """Computes the first moments p^eps_k of the the equilibrium densities sigma^eps_k:

        p^eps_k = int_{Gamma^eps_k} (y - c_k) . sigma_k(y) dGamma_y.

        Returns
        -------
        val : np.ndarray
            A 2D NumPy array of dimension (M, 3), where M is the number of particles,
            containing p^eps_k for each particle k.
        """
        val = np.zeros((self.M, 3), dtype=complex)
        for k, (sigma_k, c_k) in enumerate(zip(self.equilibriums, self.config.centers)):
            space_k = sigma_k.space
            for i in range(3):

                @bempp.api.complex_callable
                def second_member(x, n, domain_index, result):
                    result[0] = x[i] - c_k[i]

                mom_func = bempp.api.GridFunction(space_k, fun=second_member)
                val[k, i] = prod_grid_fun(mom_func, sigma_k)
        return val

    def _compute_moments_inc_field(self):
        """
        Compute integrals of the following function d . (y - c_k) sigma_k(y), . stands for the inner product

        Returns
        ------
        val : np.ndarray
            A 1D NumPy array of dimension (M, 1), here M is the number of particles.
        """
        val = np.empty(self.M, dtype=object)
        d = self.config.inc_field["d"]
        for k, (sigma_k, c_k) in enumerate(zip(self.equilibriums, self.config.centers)):
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

    def rhs(self, epsilon):
        """
        Computes the right-hand side of the convolutional in time system:

        q^eps_l(t) = -u^inc(c_l, t) c^eps_l - grad u^inc(c_l, t) . p^eps_l.

        Parameters
        ----------
        epsilon : float
            Scaling parameter in the range (0, 1].

        Returns
        -------
        val : np.ndarray
            A complex-valued 2D NumPy array of dimension (N+1, M), where N+1 is the
            number of time steps and M is the number of particles.
        """
        self._ensure_initialized()

        val = np.empty((self.N + 1, self.M), dtype=complex)

        u_inc = define_incident_field(**self.config.inc_field)
        grad_u_inc = define_incident_field_grad(**self.config.inc_field)

        for k, (cap_k, pot_k, c_k) in enumerate(
            zip(self.capacitances, self.first_moments_inc_field, self.config.centers)
        ):
            val[:, k] = (
                epsilon * u_inc(c_k[0], c_k[1], c_k[2], self.t) * cap_k
                + epsilon**2 * grad_u_inc(c_k[0], c_k[1], c_k[2], self.t) * pot_k
            )
        return val

    def galerkin_matrix(self, omega, epsilon):
        """
        Computes the Laplace-domain Galerkin matrix M^eps(omega) analytically.

        The entries are calculated using the capacitances c^eps_k and
        the spherical Bessel (j_0), Hankel (h_0) functions.

        Parameters
        ----------
        omega : complex
            Complex frequency (Laplace transform parameter).
        epsilon : float
            Scaling parameter in the range (0, 1].

        Returns
        -------
        val : np.ndarray
            A complex-valued 2D NumPy array of dimension (M, M), where M is
            the number of particles.
        """
        self._ensure_initialized()

        radiis = epsilon * self.capacitances / (4 * np.pi)
        val = np.empty((self.M, self.M), dtype=complex)
        # Diagonal entries
        for k, (r_k, c_k) in enumerate(zip(radiis, self.config.centers)):
            val[k, k] = (
                4
                * 1j
                * np.pi
                * r_k**2
                * omega
                * j0(omega * r_k)
                * h0(omega * r_k)
                * np.exp(abs(np.imag(omega * r_k)) + 1j * omega * r_k)
            )
        # Off-diagonal entries
        for k, (r_k, c_k) in enumerate(zip(radiis, self.config.centers)):
            for l in range(k + 1, self.M):
                r_l, c_l = radiis[l], self.config.centers[l]
                dist = np.linalg.norm(np.array(c_k) - np.array(c_l))
                val[k, l] = (
                    4
                    * 1j
                    * np.pi
                    * omega
                    * r_k
                    * r_l
                    * j0(omega * r_k)
                    * j0(omega * r_l)
                    * h0(omega * dist)
                    * np.exp(
                        abs(np.imag(omega * r_l))
                        + abs(np.imag(omega * r_k))
                        + 1j * omega * dist
                    )
                )
                val[l, k] = val[k, l]
        return val

    def density(self, epsilon):
        """
        Computes the unknown density coefficients lambda^eps_k(t) by solving
        the convolutional-in-time system.

        This method solves the following system of convolutional equations:

            sum_{k=1}^N (K^eps_lk * lambda^eps_k)(t) = q^eps_l(t), l=1,...,M

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
        self._ensure_initialized()
        G = self.rhs(epsilon)
        A = lambda s: self.galerkin_matrix(s, epsilon)
        val = cq(A, G, self.config.dt, self.rho, self.N, self.config.multistep_method)
        return val

    def potential(self, pot, cap, epsilon, omega, c):
        """
        Evaluates an approximated single-layer potential operator for the 3D Helmholtz problem.

        Instead of direct surface integration, this method uses first two term of
        the Taylor's expansionof the kernel around the particle center 'c'.
        This approach avoids heavynumerical integration by using precomputed
        capacitances (cap) and first moments (pot).

        Parameters
        ----------
        epsilon : float
            Scaling parameter.
        cap : float
            Capasitance.
        pot : np.ndarray
            First moment.
        omega : complex
            Complex frequency (Fourier-Lplace parameter).
        c : np.ndarray
            Center of the particle.

        Returns
        -------
        val : complex
            The complex-valued potential evaluated at the observation point x_0.
        """
        x_0 = self.config.x_0.flatten()
        c_vec = np.array(c).flatten()
        dist = np.linalg.norm(c_vec - x_0)

        inner_prod = np.dot(c_vec - x_0, pot)

        G = helmholtz_greens_function_3d(omega, dist)
        val = G * (
            epsilon * cap + epsilon**2 * (1j * omega / dist - 1 / dist**2) * inner_prod
        )
        return val

    def compute_scattered_field(self, epsilon):
        """
        Computes the approximated scattered field u^eps_app(x_0, t) at a single point.

        This method reconstructs the time-domain scattered field by summing the
        contributions from all M particles using the Convolution Quadrature (CQ) method:

        u^eps_app(x, t) = sum_{k=1}^N lambda^eps_k(t-|x-c_k|)/ (4 pi |x-c_k|) (c^eps_k + (x-c_k) · p^eps_k/|x-c_k|^2)
                        + sum_{k=1}^N partial_t lambda^eps_k(t-|x-c_k|) / (4 pi |x-c_k|^2) (x-c_k) · p^eps_k.

        Parameters
        ----------
        epsilon : float
            Scaling parameter in the range (0, 1].

        Returns
        -------
        val : np.ndarray
            A complex-valued 1D NumPy array of dimension (N+1,) representing
            the scattered field at point x_0 for each time step.
        """

        self._ensure_initialized()

        J = self.density(epsilon)
        val = 0

        for k, (c_k, cap_k, pot_k) in enumerate(
            zip(self.config.centers, self.capacitances, self.first_moments)
        ):
            A = lambda s: self.potential(pot_k, cap_k, epsilon, s, c_k)
            val += cq_potential(
                A,
                J[:, k],
                self.config.dt,
                self.rho,
                self.N,
                self.config.multistep_method,
            )
        return val

    def potential_plane(self, pot, cap, epsilon, omega, c, points, idx):
        """
        Evaluates the approximated Helmholtz potential over a grid of points.

        Parameters
        ----------
        pot : np.ndarray
            First moment.
        cap : float
            Capacitance.
        epsilon : float
            Scaling parameter.
        omega : complex
            Complex frequency (Fourier-Laplace parameter).
        c : np.ndarray
            Center of the particle.
        points : np.ndarray
            Array of coordinates (3, P) representing the evaluation grid.
        idx : np.ndarray
            Boolean mask identifying points located outside the particles.

        Returns
        -------
        val : np.ndarray
            A 2D array (P, 1) containing the potential evaluated at the grid points.
        """

        val = np.zeros((points.shape[0], 1), dtype=np.complex128)

        X = points[idx, :]  # (P^2, 3)
        c_vec = np.array(c).reshape(1, 3)  # (1, 3)

        diff = X - c_vec
        dist = np.linalg.norm(diff, axis=1).reshape(-1, 1)

        inner_prod = (c_vec - X) @ pot.reshape(3, 1)

        G = helmholtz_greens_function_3d(omega, dist)

        res = (
            epsilon * cap * G
            + epsilon**2 * (1j * omega / dist - 1 / dist**2) * inner_prod * G
        )

        val[idx, 0] = res.ravel()
        return val

    def compute_scattered_field_plane(self, epsilon):
        """
        Computes the scattered field u^eps_app on a 2D grid for all time steps.

        This method evaluates the total scattered field over a spatial plane.
        Convolution quadrature is applied to reconstruct the time-domain signal
        at each grid point.

        Points located inside the particles are masked with np.nan.

        Parameters
        ----------
        epsilon : float
            Scaling parameter in the range (0, 1].

        Returns
        -------
        val : np.ndarray
            A 3D complex-valued array of dimension (N+1, N_grid, N_grid)
            representing the time evolution of the scattered field on a plane.
        """
        self._ensure_initialized()

        xmin, xmax, ymin, ymax = self.config.limits

        points, idx = get_grid(
            self.config.geometry,
            self.config.centers,
            self.config.N_grid,
            xmin,
            xmax,
            ymin,
            ymax,
            epsilon=epsilon,
            plane=self.config.plane,
        )
        points = points.T

        J = self.density(epsilon)
        val = np.zeros(
            (self.N + 1, self.config.N_grid, self.config.N_grid), dtype=complex
        )

        for k, (c_k, cap_k) in enumerate(zip(self.config.centers, self.capacitances)):
            pot_k = self.first_moments[k, :]

            A = lambda s: self.potential_plane(
                pot_k, cap_k, epsilon, s, c_k, points, idx
            )

            val += cq_potential_plane(
                A,
                J[:, k],
                self.config.dt,
                self.rho,
                self.N,
                self.config.multistep_method,
                self.config.N_grid,
            )

        mask_2d = idx.reshape(self.config.N_grid, self.config.N_grid)
        val[:, ~mask_2d] = np.nan

        return val


def prod_grid_fun(f, g):
    """
    Computes the L2 inner product of two Bempp GridFunctions.

    This function calculates the integral of the product of functions f and g
    over the boundary Gamma:

    val = int_{Gamma} f(x) g(x) dGamma_x

    Parameters
    ----------
    f : bempp.api.GridFunction
        The first grid function.
    g : bempp.api.GridFunction
        The second grid function.

    Returns
    -------
    val : complex
        The scalar value of the L2 inner product.
    """
    val = f.projections(g.space) @ g.coefficients
    return val
