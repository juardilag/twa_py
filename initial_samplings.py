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

@jax.jit(static_argnames=('coupling_type',))
def discrete_spin_sampling_factorized(key, target_vector, coupling_type):
    # ... your existing logic ...
    n = target_vector / (jnp.linalg.norm(target_vector) + 1e-12)
    
    # Probabilities
    p_y = jnp.clip((1.0 + n[1]) / 2.0, 0.0, 1.0)
    p_z = jnp.clip((1.0 + n[2]) / 2.0, 0.0, 1.0)
    
    k1, k2, k3 = jax.random.split(key, 3)
    
    if coupling_type == "x_coupling":
        # Deterministic projection logic
        sx = n[0] 
        sy = jnp.where(jax.random.uniform(k2) < p_y, 1.0, -1.0)
        sz = jnp.where(jax.random.uniform(k3) < p_z, 1.0, -1.0)
    else:
        # Full discrete logic
        p_x = jnp.clip((1.0 + n[0]) / 2.0, 0.0, 1.0)
        sx = jnp.where(jax.random.uniform(k1) < p_x, 1.0, -1.0)
        sy = jnp.where(jax.random.uniform(k2) < p_y, 1.0, -1.0)
        sz = jnp.where(jax.random.uniform(k3) < p_z, 1.0, -1.0)
        
    return jnp.array([sx, sy, sz])


def gaussian_spin_sampling(key, target_vector):
    """
    Samples Sx, Sy, Sz from a Gaussian distribution.
    This replaces the discrete +/- 1 to eliminate the beating artifact.
    """
    n = target_vector / (jnp.linalg.norm(target_vector) + 1e-12)
    
    # In TWA, the variance of a spin component is 1 - n^2
    # This ensures the total 'length' of the spin is correct on average.
    k1, k2, k3 = jax.random.split(key, 3)
    
    sx = n[0] + jax.random.normal(k1) * jnp.sqrt(1.0 - n[0]**2)
    sy = n[1] + jax.random.normal(k2) * jnp.sqrt(1.0 - n[1]**2)
    sz = n[2] + jax.random.normal(k3) * jnp.sqrt(1.0 - n[2]**2)
    
    return jnp.array([sx, sy, sz])


def spherical_spin_sampling(key, target_vector):
    """
    Samples the spin from a distribution on the surface of the Bloch sphere.
    Eliminates Gaussian tails while keeping the continuum needed to avoid beating.
    """
    # 1. Standardize the target direction
    n = target_vector / (jnp.linalg.norm(target_vector) + 1e-12)
    
    # 2. Sample two random angles for the 'noise' cloud
    k1, k2 = jax.random.split(key)
    
    # The radius is often chosen as 1.0 (Pauli) or sqrt(3)/2
    # For matching QuTiP sigmax/y/z, radius 1.0 is standard.
    radius = 1.0 
    
    # We sample a small patch on the sphere centered at 'n'
    # A simple way is to add perpendicular Gaussian noise and re-normalize
    noise = jax.random.normal(k1, (3,)) * 0.5 # Width of the 'quantum' cloud
    s_total = n + noise
    
    # Re-normalize to the surface of the sphere
    s_sampled = (s_total / jnp.linalg.norm(s_total)) * radius
    
    return s_sampled


def projected_gaussian_sampling(key, target_vector):
    """
    Standard TWA for Spin-1/2:
    Keeps the target component fixed at 1.0.
    Adds Gaussian fluctuations to the transverse axes.
    """
    # 1. Target axis is fixed to 1.0 (Sy = 1.0)
    # This guarantees the orange line starts at 1.0
    n = target_vector / (jnp.linalg.norm(target_vector) + 1e-12)
    
    # 2. Transverse Fluctuations (Sx and Sz)
    # We need <Sx^2> = 1.0 to match the Pauli variance <sigma_x^2> = 1
    # This provides the exact decay rate seen in QuTiP
    k1, k2 = jax.random.split(key)
    sx_noise = jax.random.normal(k1) * 1.0
    sz_noise = jax.random.normal(k2) * 1.0
    
    # Assuming target is [0, 1, 0]:
    return jnp.array([sx_noise, n[1], sz_noise])

@jax.jit
def sample_coherent_discrete_rings(
    key: jax.Array,
    alpha_mean: complex,
    use_wigner_radius: bool = True 
) -> complex:
    """
    Samples a 'Quantized' Coherent state for DTWA.
    Maps the Poissonian nature of light to discrete Fock-space rings.
    """
    k1, k2 = jax.random.split(key)
    
    # 1. Mean photon number
    mean_n = jnp.abs(alpha_mean)**2
    
    # 2. Sample Integer Photon Number n ~ Poisson(|alpha|^2)
    # This captures the 'Shot Noise' of the coherent state
    n_integer = jax.random.poisson(k1, mean_n)
    
    # 3. Calculate Radius (Wigner shift)
    # The +0.5 is vital for the Fluctuation-Dissipation Theorem balance
    offset = jnp.where(use_wigner_radius, 0.5, 0.0)
    radius = jnp.sqrt(n_integer + offset)
    
    # 4. Sample Random Phase
    # This maintains the U(1) symmetry of the state
    phase = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
    
    # 5. Construct Complex Alpha
    # We shift the phase by the angle of alpha_mean to align the 'blob'
    alpha_sample = radius * jnp.exp(1j * (phase + jnp.angle(alpha_mean)))
    
    return alpha_sample