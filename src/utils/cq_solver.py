import numpy as np

def delta(rho, k, N, method):
    """Computes the generating function for the multistep method."""
    zeta = rho * np.exp(-2j * np.pi * k / (N + 1))
    
    if method == 'BDF1':
        return 1 - zeta
    elif method == 'BDF2':
        return 1.5 - 2 * zeta + 0.5 * zeta**2
    elif method == 'TR':
        return 2 * (1 - zeta) / (1 + zeta)
    elif method == 'TTR':
        ck = np.array([0.893817850529318, 0.684154908023834, 0.629642997466429])
        val = 1 - zeta + 0.5 * (1 - zeta)**2 
        for i, c in enumerate(ck, start=2):
            val += 2**(-i) * c * (1 - zeta)**(i + 1)
        return val
    raise ValueError(f"Method {method} not supported")

def _cq_core_logic(G, rho, N):
    """
    Internal helper to handle shared CQ logic: scaling, FFT, and inverse scaling.
    """
    l = np.arange(N + 1)
    R = rho**l
    R_inv = rho**(-l)
    
    G_scaled = (G.T * R).T
    G_fft = np.fft.fft(G_scaled, axis=0)
    
    return l, G_fft, R_inv

def cq(A, G, dt, rho, N, method):
    """Standard CQ solver for linear systems."""
    l, G_fft, R_inv = _cq_core_logic(G, rho, N)
    M = G.shape[1]
    K = np.empty((N + 1, M), dtype=complex)
    
    for n in l:
        s = 1j * delta(rho, n, N, method) / dt
        K[n, :] = np.linalg.solve(A(s), G_fft[n, :])
        
    res = np.fft.ifft(K, axis=0)
    return (res.T * R_inv).T

def cq_potential(A, G, dt, rho, N, method):
    """CQ for scalar potential evaluation."""
    l, G_fft, R_inv = _cq_core_logic(G, rho, N)
    K = np.empty(N + 1, dtype=complex)
    
    for n in l:
        s = 1j * delta(rho, n, N, method) / dt
        K[n] = A(s) * G_fft[n]
        
    res = np.fft.ifft(K)
    return res * R_inv

def cq_potential_plane(A, G, dt, rho, N, method, N_grid): 
    """CQ for 2D plane potential evaluation."""
    l, G_fft, R_inv = _cq_core_logic(G,rho, N)
    
    K_matrices = np.empty((N + 1, N_grid, N_grid), dtype=complex)
    
    for n in l:
        s = 1j * delta(rho, n, N, method) / dt
        K_matrices[n] = A(s).reshape(N_grid, N_grid)
    
    K_matrices *= G_fft[:, np.newaxis, np.newaxis]
    
    res = np.fft.ifft(K_matrices, axis=0)
    
    return res * R_inv[:, np.newaxis, np.newaxis]