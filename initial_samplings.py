import jax.numpy as jnp
import jax

def discrete_spin_sampling(
    key, 
    n_trajectories : int, 
    initial_z : int = -1.0):
    """
    Generates N initial spin vectors using Discrete sampling.
    
    Args:
        key: JAX PRNGKey for reproducibility.
        n_trajectories: Number of trajectories to simulate (N).
        initial_z: The initial polarization of the spin (-1.0 for Down, +1.0 for Up).
        
    Returns:
        s_init: Array of shape (n_trajectories, 3).
    """
    # 1. Split the random key to generate independent noise for x and y
    k1, k2 = jax.random.split(key)
    
    # 2. Sample Transverse Components (Quantum Fluctuations)
    # We use Bernoulli sampling to get 0 or 1, then map: 0 -> -1, 1 -> +1
    # Formula: 2 * (0 or 1) - 1 = (-1 or +1)
    sx = 2.0 * jax.random.bernoulli(k1, p=0.5, shape=(n_trajectories,)).astype(jnp.float32) - 1.0
    sy = 2.0 * jax.random.bernoulli(k2, p=0.5, shape=(n_trajectories,)).astype(jnp.float32) - 1.0
    
    # 3. Set Longitudinal Component (Mean Field value)
    # This is fixed for all trajectories (no fluctuations in the eigenbasis)
    sz = jnp.full((n_trajectories,), initial_z)
    
    # 4. Stack into a single matrix (N, 3)
    s_init = jnp.stack([sx, sy, sz], axis=1)
    
    return s_init


def continious_spin_sampling(
    key, 
    n_traj : int):
    """
    Generates N initial spin vectors using gaussian sampling.
    
    Args:
        key: JAX PRNGKey for reproducibility.
        n_trajectories: Number of trajectories to simulate (N).
    Returns:
        s_init: Array of shape (n_trajectories, 3).
    """
    k1, k2 = jax.random.split(key)
    # Normal(0, 1) to match variance <sigma^2>=1
    sx = jax.random.normal(k1, (n_traj,)) 
    sy = jax.random.normal(k2, (n_traj,))
    # Fixed mean for z (approximate, ignores longitudinal noise)
    sz = jnp.full((n_traj,), -1.0)
    return jnp.stack([sx, sy, sz], axis=1)