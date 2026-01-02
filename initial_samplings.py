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


def boson_sampling(key, n_trajectories, initial_alpha=0.0):
    """
    Generates N initial boson states (complex scalars) using Gaussian Wigner sampling.
    Correct for Coherent States |alpha> and Vacuum |0>.
    
    Args:
        key: JAX PRNGKey.
        n_trajectories: Number of trajectories.
        initial_alpha: The complex mean field amplitude (default 0.0 for Vacuum).
        
    Returns:
        a_init: Complex Array of shape (n_trajectories,).
    """
    k1, k2 = jax.random.split(key)
    
    # 1. Sample standard normal noise N(0, 1)
    # We need the Wigner width for vacuum: <|delta_a|^2> = 1/2.
    # This means Real and Imag parts each need Variance = 1/4.
    # Standard Deviation = sqrt(1/4) = 0.5.
    
    noise_real = 0.5 * jax.random.normal(k1, shape=(n_trajectories,))
    noise_imag = 0.5 * jax.random.normal(k2, shape=(n_trajectories,))
    
    # 2. Combine into complex fluctuations
    # The factor 0.5 ensures that <|noise|^2> = 0.5 (Half photon energy)
    noise = noise_real + 1j * noise_imag
    
    # 3. Add to the mean field value
    a_init = initial_alpha + noise
    
    return a_init