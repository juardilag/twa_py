import numpy as np
from qutip import *

def solve_dynamics(Bx, By, Bz, wa, g, kappa, times, rho0, N):
    # 1. Operators
    a = tensor(qeye(2), destroy(N))
    sx, sy, sz = tensor(sigmax(), qeye(N)), tensor(sigmay(), qeye(N)), tensor(sigmaz(), qeye(N))
    
    # 2. Hamiltonian (Arbitrary B)
    # Factor 0.5 assumes gamma=1 for spin-1/2
    H = 0.5*(Bx*sx + By*sy + Bz*sz) + wa * a.dag() * a + g * sx * (a + a.dag())
    
    # 3. Collapse operator (Lossy boson)
    c_ops = [np.sqrt(kappa) * a]
    
    # 4. Solve Master Equation
    # result.states contains rho(t) for every point in 'times'
    result = mesolve(H, rho0, times, c_ops, [sx, sy, sz, a.dag()*a])
    
    return result

def get_initial_state(v, N):
    v = np.array(v, dtype=float)
    n = v / np.linalg.norm(v)
    
    theta = np.arccos(n[2])
    phi = np.arctan2(n[1], n[0])
    
    # Manual ket construction
    psi_spin = (np.cos(theta/2) * basis(2, 0) + 
                np.exp(1j * phi) * np.sin(theta/2) * basis(2, 1))
    
    # Tensor with boson vacuum
    return tensor(psi_spin, basis(N, 0))