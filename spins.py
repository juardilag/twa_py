import jax.numpy as jnp
import jax

def hamiltonian(
    s : jnp.ndarray,
    t : None,
    Omega : float):
    """
    Defines the classical Hamiltonian H(s) = Omega * s_x.
    
    Args:
        s: Spin vector of shape (3,). s[0]=x, s[1]=y, s[2]=z.
        t: Time (unused for static drive, but required for ODE solvers).
        Omega: Rabi frequency.
    
    Returns:
        Scalar energy value.
    """
    return Omega * s[0]

@jax.jit
def spin_eom(
    s : jnp.ndarray,
    t : None, 
    Omega : float):
    """
    Computes the time derivative ds/dt using the Spin Poisson Bracket.
    Formula: ds/dt = {s, H} = 2 * (grad_H x s)
    
    Args:
        s: Spin vector (3,).
        t: Time.
        Omega: System parameter passed to Hamiltonian.
         
    Returns:
        ds_dt: Vector of shape (3,) representing velocity in phase space.
    """
    # 1. Calculate Gradient of Hamiltonian
    # jax.grad automatically computes (dH/ds_x, dH/ds_y, dH/ds_z)
    dH_ds = jax.grad(hamiltonian, argnums=0)(s, t, Omega)
    
    # 2. Apply the Spin Precession Rule
    # The factor of 2.0 comes from the Pauli algebra {s_i, s_j} = 2*epsilon*s_k
    ds_dt = 2.0 * jnp.cross(dH_ds, s)
    
    return ds_dt


