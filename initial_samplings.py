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

def discrete_spin_sampling_factorized(key, initial_direction, n_spins=1):
    """
    Tu lógica original generalizada para un modelo colectivo de N espines.
    S(0) = N * Mean + F1 * perp1 + F2 * perp2
    Las fluctuaciones transversales F1 y F2 son la suma de N variables +/- 1.
    """
    k1, k2 = jax.random.split(key)
    
    # 1. Dirección media
    mean_vec = initial_direction / (jnp.linalg.norm(initial_direction) + 1e-12)
    
    # 2. Gram-Schmidt para encontrar ejes perpendiculares
    # Usamos jnp.where en lugar de if para que sea compatible con JIT
    v = jnp.where(jnp.abs(mean_vec[0]) < 0.9, 
                  jnp.array([1.0, 0.0, 0.0]), 
                  jnp.array([0.0, 1.0, 0.0]))
    
    perp1 = jnp.cross(mean_vec, v)
    perp1 = perp1 / (jnp.linalg.norm(perp1) + 1e-12)
    perp2 = jnp.cross(mean_vec, perp1)
    
    # 3. Fluctuaciones discretas +/- 1 para N espines
    # Generamos N variables aleatorias para cada eje transversal y las sumamos
    flips1 = 2.0 * jax.random.bernoulli(k1, p=0.5, shape=(n_spins,)) - 1.0
    flips2 = 2.0 * jax.random.bernoulli(k2, p=0.5, shape=(n_spins,)) - 1.0
    
    f1 = jnp.sum(flips1)
    f2 = jnp.sum(flips2)
    
    # 4. Construcción del espín macroscópico
    # El campo medio escala linealmente con N, las fluctuaciones escalan como sqrt(N)
    s_init = (n_spins * mean_vec) + (f1 * perp1) + (f2 * perp2)
    
    return s_init


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