import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, sin, cos, exp

x = symbols('x')
f = lambda x: sin(4*x**2)*x*exp(-x)
f_derivative = diff(f(x), x)

f_prime = lambda t: float(f_derivative.subs(x, t))

dt = 0.1
T = 16
N = int(T/dt)
t = np.arange(0, T+dt, dt)

g = np.zeros(np.shape(t))

for n, (t_n) in enumerate(t):
    K = int(t_n // 2)
    g[n] = 2*sum(f_prime(t_n - 2*k) for k in range(1, K+1))