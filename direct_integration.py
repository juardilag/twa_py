import jax.numpy as jnp
from jax import vmap
import jax
from initial_samplings import discrete_spin_sampling

def get_spectral_density(omega, eta, omega_c, s):
    # J(omega) = eta * omega^s * omega_c^(1-s) * exp(-omega/omega_c)
    safe_omega = jnp.where(omega > 0, omega, 1.0) 
    
    # Using jnp.power once is more efficient
    J_val = eta * safe_omega * jnp.power(safe_omega / omega_c, s - 1) * jnp.exp(-safe_omega / omega_c)
    
    return jnp.where(omega > 0, J_val, 0.0)


def get_A_omega(omega, eta, omega_c, s):
    """Spectral function A(omega) = 2 * J(omega) for positive omega."""
    return 2.0 * get_spectral_density(omega, eta, omega_c, s)


def compute_memory_kernel(tau_grid, eta, omega_c, s, g=1.0, num_omega=10_000):
    """
    Computes gamma(tau) = g^2 * integral(A(omega)/pi * sin(omega*tau) d_omega)
    """
    # Define an integration grid for omega
    # We go up to 10*omega_c to ensure the exponential cutoff has acted
    omega_max = 100*omega_c
    omega_grid = jnp.linspace(0, omega_max, num_omega)
    d_omega = omega_grid[1] - omega_grid[0]
    
    A_vals = get_A_omega(omega_grid, eta, omega_c, s)
    
    def single_tau_kernel(t):
        # The integral from your notes
        integrand = (A_vals / jnp.pi) * jnp.sin(omega_grid * t)
        return g**2 * jnp.sum(integrand)*d_omega

    # Vmap over the time grid to get the kernel series
    return vmap(single_tau_kernel)(tau_grid)


def generate_noise(key, t_grid, eta, omega_c, s, kBT, num_omega=1000):
    """
    Generates a time-series of noise xi(t) for a single trajectory.
    """
    # 1. Setup frequency grid for the bath
    omega_max = 10.0 * omega_c
    omega_grid = jnp.linspace(0.01, omega_max, num_omega) # Avoid omega=0
    d_omega = omega_grid[1] - omega_grid[0]
    
    # 2. Get Spectral Function A(omega)
    # A(omega) = 2 * eta * omega^s * omega_c^(1-s) * exp(-omega/omega_c)
    A_omega = 2.0 * eta * omega_grid * jnp.power(omega_grid/omega_c, s-1) * jnp.exp(-omega_grid/omega_c)
    
    # 3. Sample Stochastic Processes (Initial Conditions)
    key_a, key_b = jax.random.split(key)
    # 3 components for the vector noise (x, y, z)
    A_stoch = jax.random.normal(key_a, (num_omega, 3)) 
    B_stoch = jax.random.normal(key_b, (num_omega, 3))
    
    # 4. Amplitude weighting from your notes: sqrt( (kBT * A(omega) * d_omega) / (pi * omega^2) )
    amplitude = jnp.sqrt((kBT * A_omega * d_omega) / (jnp.pi * omega_grid**2))
    
    # 5. Construct xi(t) for all t in t_grid
    # We use broadcasting to compute (num_t, num_omega)
    cos_term = jnp.cos(jnp.outer(t_grid, omega_grid)) # (T, W)
    sin_term = jnp.sin(jnp.outer(t_grid, omega_grid)) # (T, W)
    
    # xi_t = sum_w amplitude * (A_stoch * cos + B_stoch * sin)
    # Final shape: (num_t, 3)
    xi_t = (cos_term @ (amplitude[:, None] * A_stoch) + 
            sin_term @ (amplitude[:, None] * B_stoch))
    
    return xi_t


def n_markovian_step(state, step_idx, noise_traj, gamma_kernel, B_field, g, dt):
    """
    JAX-compatible step using masking to avoid dynamic slicing errors.
    """
    # 1. Get current spin
    S_t = state[step_idx - 1]
    
    # 2. Compute Memory Integral using a Mask
    # Create a mask that is 1.0 for indices < step_idx and 0.0 otherwise
    # This keeps the array size static for JIT
    num_steps = state.shape[0]
    mask = (jnp.arange(num_steps) < step_idx).astype(jnp.float32)
    
    # Compute the full convolution kernel shifted to the current time
    # We use jnp.roll to align gamma(t - t') with S(t')
    # Or more simply: use the mask on a reversed kernel
    relevant_kernel = jnp.roll(gamma_kernel[::-1], step_idx)
    
    # Weighted sum over the WHOLE buffer, but mask out the 'future'
    memory_val = jnp.sum(
        (relevant_kernel[:, None] * state) * mask[:, None], 
        axis=0
    ) * dt
    
    # 3. Effective Field
    xi_t = noise_traj[step_idx - 1]
    total_field = B_field + xi_t + (g**2) * memory_val [cite: 54]
    
    # 4. Evolution (Heun Step)
    dSdt_1 = jnp.cross(S_t, total_field)
    S_inter = S_t + dSdt_1 * dt
    
    xi_next = noise_traj[step_idx]
    total_field_next = B_field + xi_next + (g**2) * memory_val 
    dSdt_2 = jnp.cross(S_inter, total_field_next)
    
    S_next = S_t + 0.5 * (dSdt_1 + dSdt_2) * dt
    
    # 5. Normalization to preserve spin magnitude
    S_next = S_next / jnp.linalg.norm(S_next)
    
    return state.at[step_idx].set(S_next), S_next


