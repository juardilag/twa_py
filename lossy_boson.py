import numpy as np
from qutip import (tensor, qeye, destroy, jmat, spin_coherent, 
                   fock_dm, coherent_dm, mesolve)

def solve_dynamics_vacuum(Bx, By, Bz, wa, g, kappa, times, rho0, N_spins=1, N_boson=30):
    """
    Solves the exact Quantum Master Equation for the Dicke model using QuTiP.
    Uses the collective spin basis to scale efficiently with N_spins.
    """
    # Total spin quantum number
    S = N_spins / 2.0
    dim_spin = int(2 * S + 1)
    
    # 1. Define Operators in the joint Hilbert space
    # jmat(S, 'x') returns the macroscopic spin operator Sx
    a = tensor(qeye(dim_spin), destroy(N_boson))
    Sx = tensor(jmat(S, 'x'), qeye(N_boson))
    Sy = tensor(jmat(S, 'y'), qeye(N_boson))
    Sz = tensor(jmat(S, 'z'), qeye(N_boson))
    
    # 2. Construct the Hamiltonian components
    # jmat already represents S = sigma/2, so the 0.5 factor is built-in
    H_spin = Bx * Sx + By * Sy + Bz * Sz
    H_boson = wa * a.dag() * a
    
    # Thermodynamic scaling: 2g/sqrt(N) * Sx * (a + a^dag)
    # For N=1, this naturally recovers g * sigma_x * (a + a^dag)
    g_scaled = (2.0 * g) / np.sqrt(N_spins)
    H_int = g_scaled * Sx * (a + a.dag())
    
    H = H_spin + H_boson + H_int
    
    # 3. Define Lindblad collapse operators (lossy boson)
    c_ops = [np.sqrt(kappa) * a]
    
    # 4. Observables to track: <Sx>, <Sy>, <Sz>, and photon number
    e_ops = [Sx, Sy, Sz, a.dag() * a]
    
    # 5. Evolve the system
    result = mesolve(H, rho0, times, c_ops, e_ops)
    return result

def get_initial_state(v, n_photons, N_spins=1, N_boson=30):
    """
    Constructs the initial density matrix rho(0) = rho_spin (x) rho_boson 
    for N spins and a cavity.
    """
    S = N_spins / 2.0
    dim_spin = int(2 * S + 1)
    
    # 1. Spin State (Spin Coherent State on the Bloch sphere)
    v = np.array(v, dtype=float)
    norm_v = np.linalg.norm(v)
    
    # Safeguard against zero-division if a null vector is passed
    if norm_v < 1e-12:
        n_vec = np.array([0.0, 0.0, 1.0]) 
    else:
        n_vec = v / norm_v
        
    theta = np.arccos(n_vec[2])
    phi = np.arctan2(n_vec[1], n_vec[0])
    
    # QuTiP's built-in function to generate a macroscopic coherent spin state
    psi_spin = spin_coherent(S, theta, phi)
    
    # Convert pure state to density matrix
    rho_spin = psi_spin * psi_spin.dag()

    # 2. Boson State
    if n_photons == 0:
        # Exact vacuum state matches the T=0 constraint 
        rho_boson = fock_dm(N_boson, 0) 
    else:
        # True Coherent state |alpha><alpha| with alpha = sqrt(n)
        rho_boson = coherent_dm(N_boson, np.sqrt(n_photons))
        
    # Return the joint state in the tensor product space
    return tensor(rho_spin, rho_boson)