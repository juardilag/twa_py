import numpy as np
from qutip import *

def solve_dynamics_vacuum(Bx, By, Bz, wa, g, kappa, times, rho0, N=15):
    """
    Solves the Master Equation for T=0.
    Aligned with the derivation: (d/dt + i*wa + kappa/2)a = -i * g * sx
    """
    # 1. Operators in the Composite Hilbert Space
    a = tensor(qeye(2), destroy(N))
    sx, sy, sz = tensor(sigmax(), qeye(N)), tensor(sigmay(), qeye(N)), tensor(sigmaz(), qeye(N))
    
    # 2. Hamiltonian Construction
    H_spin = 0.5 * (Bx * sx + By * sy + Bz * sz)
    H_boson = wa * a.dag() * a
    H_int = g * sx * (a + a.dag())
    
    H = H_spin + H_boson + H_int
    
    c_ops = [np.sqrt(kappa) * a]
    e_ops = [sx, sy, sz, a.dag() * a]
    result = mesolve(H, rho0, times, c_ops, e_ops)
    
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