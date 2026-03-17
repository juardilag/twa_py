import numpy as np
from qutip import *

def solve_dynamics_thermal(Bx, By, Bz, wa, g, kappa, kBT, times, rho0, N=15):
    # 1. Operators
    a = tensor(qeye(2), destroy(N))
    sx, sy, sz = tensor(sigmax(), qeye(N)), tensor(sigmay(), qeye(N)), tensor(sigmaz(), qeye(N))
    
    # 2. Hamiltonian (Note the -0.5 sign to match your analytical notes exactly!)
    H = 0.5*(Bx*sx + By*sy + Bz*sz) + wa * a.dag() * a + g * sx * (a + a.dag())
    
    # 3. Thermal Bath Calculations
    if kBT == 0.0:
        n_th = 0.0
    else:
        # Bose-Einstein distribution for thermal photons
        n_th = 1.0 / (np.exp(wa / kBT) - 1.0)
        
    # 4. Thermal Collapse Operators (Detailed Balance)
    c_decay = np.sqrt(kappa * (1.0 + n_th)) * a      # Spontaneous + Stimulated Emission
    c_pump  = np.sqrt(kappa * n_th) * a.dag()        # Thermal Absorption
    
    c_ops = [c_decay, c_pump]
    
    # 6. Solve Master Equation
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