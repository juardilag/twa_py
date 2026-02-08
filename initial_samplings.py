import jax.numpy as jnp
import jax

def discrete_spin_sampling(key, n_trajectories, initial_direction=jnp.array([0.0, 0.0, 1.0])):
    """
    Generates N initial spin vectors for TWA.
    The mean spin points along 'initial_direction', with discrete fluctuations (+/-1)
    in the two perpendicular axes.
    """
    k1, k2 = jax.random.split(key)
    
    # 1. Ensure initial_direction is a unit vector
    mean_vec = initial_direction / jnp.linalg.norm(initial_direction)
    
    # 2. Find two perpendicular vectors (Gram-Schmidt style)
    # We pick an arbitrary vector 'v' to start the process
    v = jnp.array([1.0, 0.0, 0.0]) if jnp.abs(mean_vec[0]) < 0.9 else jnp.array([0.0, 1.0, 0.0])
    perp1 = jnp.cross(mean_vec, v)
    perp1 = perp1 / jnp.linalg.norm(perp1)
    perp2 = jnp.cross(mean_vec, perp1)
    
    # 3. Sample discrete fluctuations +/- 1 for the two perpendicular directions
    f1 = 2.0 * jax.random.bernoulli(k1, p=0.5, shape=(n_trajectories,)) - 1.0
    f2 = 2.0 * jax.random.bernoulli(k2, p=0.5, shape=(n_trajectories,)) - 1.0
    
    # 4. Construct S(0) = Mean + Fluctuation1*perp1 + Fluctuation2*perp2
    # This ensures <S(0)> = initial_direction
    s_init = (jnp.outer(jnp.ones(n_trajectories), mean_vec) + 
              jnp.outer(f1, perp1) + 
              jnp.outer(f2, perp2))
    
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

def discrete_spin_sampling_single(key, initial_direction, width_scale=0.0):
    """
    Samples discrete spins for TWA.
    
    CRITICAL CHANGE: This function does NOT normalize the output vector.
    - If width_scale=0.0 (Mean Field): Returns vector of length 1.0.
    - If width_scale=1.0 (Full TWA): Returns vector of length sqrt(3).
      This preserves the component values (+/- 1) to ensure correct 
      Rabi frequencies in non-Markovian models.
    """
    k1, k2 = jax.random.split(key)
    
    # 1. Normalize the Mean Vector (The "Center" of your blob)
    # This ensures the 'mean' part is strictly length 1.0
    mean_vec = initial_direction / (jnp.linalg.norm(initial_direction) + 1e-12)
    
    # 2. Gram-Schmidt for perpendicular directions
    is_y_aligned = jnp.abs(mean_vec[1]) > 0.9
    ref_vec = jnp.where(is_y_aligned, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
    perp1 = jnp.cross(mean_vec, ref_vec)
    perp1 = perp1 / (jnp.linalg.norm(perp1) + 1e-12)
    perp2 = jnp.cross(mean_vec, perp1)
    
    # 3. Discrete Fluctuations (+/- 1)
    # When width_scale=1.0, these are exactly +1 or -1
    f1 = width_scale * (2.0 * jax.random.bernoulli(k1, p=0.5) - 1.0)
    f2 = width_scale * (2.0 * jax.random.bernoulli(k2, p=0.5) - 1.0)
    
    # 4. Construct the Vector
    # This places the spin on the "corners of a cube" rather than a sphere.
    raw_vec = mean_vec + f1 * perp1 + f2 * perp2
    
    # 5. RETURN WITHOUT NORMALIZATION
    # We return the "long" vector to preserve the correct eigenvalues (+/- 1).
    return raw_vec