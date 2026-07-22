import numpy as np

def exact_solotion(dt,T):
    N = int(T/dt)
    g = lambda t: t**4*np.exp(-2*t)
    t = np.arange(0, T + dt, dt)
    g_der = lambda t: (4*t**3 - 2*t**4) * np.exp(-2*t)
    time_values = np.arange(0, T , 2) 

    exact_sol = np.zeros(N)

    for t_0 in time_values:
        time_line = np.arange(t_0, t_0 + 2, dt)
        index_start = int(t_0 / dt)
        index_end = index_start + int(2 / dt)

        _g_der = g_der(time_line[:, np.newaxis] - np.arange(0, t_0 + 1, 2))

        exact_sol[index_start:index_end] += 2*_g_der.sum(axis=1)

    return g(t)