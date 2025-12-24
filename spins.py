import jax.numpy as jnp
import jax

@jax.jit
def lmg_eom(
    s : jnp.ndarray, 
    t : None, 
    args : list[float, float]):
    """
    Computes the time derivative ds/dt using the Spin Poisson Bracket.
    Formula: ds/dt = {s, H} = 2 * (grad_H x s)
    
    Args:
        s: Spin vector (3,).
        t: Time.
        args: System parameters passed to Hamiltonian.
         
    Returns:
        ds_dt: Vector of shape (3,) representing velocity in phase space.
    """
    Omega, Chi = args
    def H(state):
        # Linear Drive + Non-Linear Twisting
        return Omega * state[0] + Chi * (state[2]**2)
    
    dH_ds = jax.grad(H)(s)
    return 2.0 * jnp.cross(dH_ds, s)

@jax.jit
def linear_eom(
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
    # jax.grad automatically computes (dH/ds_x, dH/ds_y, dH/ds_z)
    def H(state):
        return Omega*state[0]    
    dH_ds = jax.grad(H)(s)
    return 2.0 * jnp.cross(dH_ds, s)


