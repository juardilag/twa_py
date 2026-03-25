import numpy as np
from qutip import *

def solve_dynamics_coherent(Bx, By, Bz, wa, g, kappa, times, rho0, N=0):
    """
    Solves the Master Equation for a driven or high-occupancy cavity.
    """
    # 1. Operators in the Composite Hilbert Space
    a = tensor(qeye(2), destroy(N))
    sx = tensor(sigmax(), qeye(N))
    sy = tensor(sigmay(), qeye(N))
    sz = tensor(sigmaz(), qeye(N))
    
    # 2. Hamiltonian Construction
    # Note: Using -0.5 to align with your LaTeX notes H = -0.5 * B * sigma
    H_spin = -0.5 * (Bx * sx + By * sy + Bz * sz)
    H_boson = wa * a.dag() * a
    H_int = g * sx * (a + a.dag())
    
    H = H_spin + H_boson + H_int
    
    # 3. Dissipation (T=0)
    c_ops = [np.sqrt(kappa) * a]
    
    # 4. Expectations: [Sx, Sy, Sz, <n>]
    e_ops = [sx, sy, sz, a.dag() * a]
    
    result = mesolve(H, rho0, times, c_ops, e_ops)
    return result

def get_initial_state_coherent(v, n_photons, N):
    """
    Creates a tensor product of a spin on the Bloch sphere 
    and a coherent state in the cavity.
    """
    # Spin part
    v = np.array(v, dtype=float)
    n = v / (np.linalg.norm(v) + 1e-12)
    
    theta = np.arccos(n[2])
    phi = np.arctan2(n[1], n[0])
    
    psi_spin = (np.cos(theta/2) * basis(2, 0) + 
                np.exp(1j * phi) * np.sin(theta/2) * basis(2, 1))
    
    # Boson part: |alpha> where |alpha|^2 = n_photons
    alpha = np.sqrt(n_photons)
    psi_boson = coherent(N, alpha)
    
    # Check truncation safety
    if n_photons > (N - 5):
        print(f"Warning: N={N} might be too small for n={n_photons} photons.")
    
    return tensor(psi_spin, psi_boson)