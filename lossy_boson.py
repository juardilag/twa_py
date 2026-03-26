import numpy as np
from qutip import (tensor, qeye, destroy, sigmax, sigmay, sigmaz, 
                   basis, fock_dm, mesolve)
from scipy.stats import poisson

def solve_dynamics_vacuum(Bx, By, Bz, wa, g, kappa, times, rho0, N=30):
    """
    Solves the exact Quantum Master Equation using QuTiP.
    Aligned directly with the system defined in the notes.
    """
    # 1. Define Operators in the joint Hilbert space
    a = tensor(qeye(2), destroy(N))
    sx = tensor(sigmax(), qeye(N))
    sy = tensor(sigmay(), qeye(N))
    sz = tensor(sigmaz(), qeye(N))
    
    # 2. Construct the Hamiltonian components
    # FIXED: Enforcing the strict -1/2 factor from H = -1/2 B * sigma
    H_spin = 0.5 * (Bx * sx + By * sy + Bz * sz)
    H_boson = wa * a.dag() * a
    H_int = g * sx * (a + a.dag())
    
    H = H_spin + H_boson + H_int
    
    # 3. Define Lindblad collapse operators (lossy boson)
    # The jump operator is derived directly from the Lindbladian
    c_ops = [np.sqrt(kappa) * a]
    
    # 4. Observables to track: <Sx>, <Sy>, <Sz>, and photon number
    e_ops = [sx, sy, sz, a.dag() * a]
    
    # 5. Evolve the system
    result = mesolve(H, rho0, times, c_ops, e_ops)
    return result

def get_initial_state(v, n_photons, N):
    """
    Constructs the initial density matrix rho(0) = rho_spin (x) rho_boson.
    Matches the function signature for drop-in execution.
    """
    # 1. Spin State (Pure state mapped from Bloch vector)
    v = np.array(v, dtype=float)
    norm_v = np.linalg.norm(v)
    
    # Safeguard against zero-division if a null vector is passed
    if norm_v < 1e-12:
        n_vec = np.array([0.0, 0.0, 1.0]) 
    else:
        n_vec = v / norm_v
        
    theta = np.arccos(n_vec[2])
    phi = np.arctan2(n_vec[1], n_vec[0])
    
    psi_spin = (np.cos(theta/2) * basis(2, 0) + 
                np.exp(1j * phi) * np.sin(theta/2) * basis(2, 1))
    
    # Convert pure state to density matrix
    rho_spin = psi_spin * psi_spin.dag()

    # 2. Boson State (Mixed Poissonian to match your rings sampling)
    if n_photons == 0:
        # Exact vacuum state matches the T=0 constraint 
        rho_boson = fock_dm(N, 0) 
    else:
        # Poissonian weight distribution for finite photons
        probs = poisson.pmf(np.arange(N), n_photons)
        probs /= probs.sum() # Normalize over truncated Fock space
        
        rho_boson = sum(p * fock_dm(N, i) for i, p in enumerate(probs))
        
    # Return the joint state in the tensor product space
    return tensor(rho_spin, rho_boson)