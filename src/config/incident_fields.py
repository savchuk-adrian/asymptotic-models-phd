import numpy as np
from numba import jit


def define_incident_field(u_inc_name, d, omega=None, sigma=None, A=None):
    """Function that defines the incident field u_inc."""
    if u_inc_name == "modulated_gaussian":

        @jit
        def second_member(x1, x2, x3, t):
            arg = d[0] * x1 + d[1] * x2 + d[2] * x3
            return np.cos(omega * (t - arg)) * np.exp(
                -sigma * (t - arg - A) ** 2
            )  #  t**4*np.exp(-2*t)

    elif u_inc_name == "sin":

        @jit
        def second_member(x1, x2, x3, t):
            arg = d[0] * x1 + d[1] * x2 + d[2] * x3
            return np.sin(omega * (t - arg))

    elif u_inc_name == "C_0_data":

        @jit
        def second_member(x1, x2, x3, t):
            arg = d[0] * x1 + d[1] * x2 + d[2] * x3
            return np.exp(-sigma * (t - arg - A) ** 2) * (t - arg - A) * (t - arg > A)

    elif u_inc_name == "plane_wave_TD":

        @jit
        def second_member(x1, x2, x3, t):
            arg = d[0] * x1 + d[1] * x2 + d[2] * x3
            return (
                np.sin(omega * (t - arg - A))
                * np.arctan(sigma * (t - arg - A))
                * (t - arg - A > 0)
            )

    elif u_inc_name == "plane_wave_sigmoid_TD":

        @jit
        def second_member(x1, x2, x3, t):
            arg = d[0] * x1 + d[1] * x2 + d[2] * x3
            sigmoid = 1 / (1 + np.exp(-sigma * (t - arg - A)))
            return np.sin(omega * (t - arg - A)) * sigmoid

    elif u_inc_name == "plane_wave":

        @jit
        def second_member(x1, x2, x3):
            arg = d[0] * x1 + d[1] * x2 + d[2] * x3
            return np.exp(1j * omega * arg)

    elif u_inc_name == "constant":

        @jit
        def second_member(x1, x2, x3):
            arg = d[0] * x1 + d[1] * x2 + d[2] * x3
            return 1

    else:
        raise ValueError(f"Unknown function `u_inc_name`: {u_inc_name}")
    return second_member


def define_incident_field_grad(u_inc_name, d, omega=None, sigma=None, A=None):
    """Defines a gradient of the incident field u_inc"""
    if u_inc_name == "modulated_gaussian":

        @jit
        def second_member_grad(x1, x2, x3, t):
            arg = d[0] * x1 + d[1] * x2 + d[2] * x3
            return (
                omega * np.sin(omega * (t - arg))
                + 2 * sigma * (t - arg - A) * np.cos(omega * (t - arg))
            ) * np.exp(-sigma * (t - arg - A) ** 2)

    elif u_inc_name == "plane_wave_sigmoid_TD":

        @jit
        def second_member_grad(x1, x2, x3, t):
            arg = t - d[0] * x1 + d[1] * x2 + d[2] * x3 - A
            sigmoid = 1 / (1 + np.exp(-sigma * arg))
            return (
                -(
                    omega * np.cos(omega * arg)
                    + np.sin(omega * arg) * sigma * (1 - sigmoid)
                )
                * sigmoid
            )

    elif u_inc_name == "C_0_data":

        @jit
        def second_member_grad(x1, x2, x3, t):
            arg = t - d[0] * x1 + d[1] * x2 + d[2] * x3 - A
            return (2 * sigma * arg - 1) * np.exp(-sigma * arg**2) * (arg > 0)

    elif u_inc_name == "plane_wave":

        @jit
        def second_member_grad(x1, x2, x3):
            arg = d[0] * x1 + d[1] * x2 + d[2] * x3
            return 1j * omega * np.exp(1j * omega * arg)

    return second_member_grad
