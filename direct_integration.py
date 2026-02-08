import jax.numpy as jnp
from jax import vmap
import jax
from initial_samplings import discrete_spin_sampling_single, discrete_spin_sampling_factorized
from tqdm.auto import tqdm

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
    omega_max = 50*omega_c
    omega_grid = jnp.linspace(1e-1, omega_max, num_omega)
    A_vals = get_A_omega(omega_grid, eta, omega_c, s)
    
    def single_tau_kernel(t):
        integrand = (A_vals / jnp.pi) * jnp.sin(omega_grid * t)
        return g**2 * jnp.trapezoid(integrand, x=omega_grid)

    return vmap(single_tau_kernel)(tau_grid)


def generate_noise(key, t_grid, eta, omega_c, s, kBT, g=1.0, num_omega=2000):
    omega_min = 1e-1  
    omega_max = 50.0 * omega_c
    
    omega_grid = jnp.linspace(omega_min, omega_max, num_omega)
    d_omega = omega_grid[1] - omega_grid[0]
    
    # 1. Spectral Function A(w) = 2 * J(w)
    safe_omega = jnp.maximum(omega_grid, 1e-12)
    A_omega = 2.0 * eta * safe_omega * jnp.power(safe_omega / omega_c, s - 1) * jnp.exp(-safe_omega / omega_c)
    
    # 2. Quantum Thermal Factor (coth)
    # coth(x) = 1/tanh(x). x = beta * w / 2
    x = safe_omega / (2.0 * kBT)
    thermal_factor = 1.0 / jnp.tanh(x)
    
    # The FDT states S_xi(w) = J(w) * coth(beta*w/2).
    # Since A(w) = 2*J(w), we have S_xi(w) = A(w)/2 * coth.
    # Total variance = Integral[ S_xi(w) * dw/pi ].
    variance_density = (A_omega * thermal_factor) / (2.0 * jnp.pi)
    
    # Amplitude for discrete sum: sqrt(2 * Variance_Density * d_omega) ?? 
    # Standard synthesis: xi(t) = Sum [ sqrt(S(w) * dw / pi) * (A cos + B sin) ] ?
    # Let's stick to the density definition: Integral (S(w)/pi) dw
    # amplitude * amplitude ~ S(w)/pi * dw
    amplitude = g * jnp.sqrt(variance_density * d_omega) 
    
    # 4. Stochastic Sampling
    key_a, key_b = jax.random.split(key)
    A_stoch = jax.random.normal(key_a, (num_omega, 3)) 
    B_stoch = jax.random.normal(key_b, (num_omega, 3))
    
    # Use simple broadcasting for time synthesis
    # shape: (num_steps, num_omega)
    cos_term = jnp.cos(jnp.outer(t_grid, omega_grid))
    sin_term = jnp.sin(jnp.outer(t_grid, omega_grid))
    
    # Sum over frequencies (axis 1)
    # (Steps, Freqs) @ (Freqs, 3) -> (Steps, 3)
    xi_t = (cos_term @ (amplitude[:, None] * A_stoch) + 
            sin_term @ (amplitude[:, None] * B_stoch))
            
    return xi_t

@jax.jit
def n_markovian_step(state, step_idx, noise_traj, gamma_kernel, B_field, dt, coupling_type="z"):
    S_t = state[step_idx - 1]
    
    # --- 1. History Convolution (Past) ---
    # Sums K(dt) ... K(t_current)
    # This misses K(0)
    indices = jnp.arange(state.shape[0])
    past_mask = (indices < step_idx)
    k_indices = (step_idx - indices)
    
    current_gamma = jnp.where(past_mask[:, None], gamma_kernel[k_indices][:, None], 0.0)
    memory_history = jnp.sum(current_gamma * state, axis=0) * dt
    
    # --- 2. Instantaneous Correction (THE FIX) ---
    # Explicitly include the self-interaction term K(0)*S(t)
    # This captures the "Spike" of the Super-Ohmic bath
    memory_instant = gamma_kernel[0] * S_t * dt
    
    # Total Memory
    memory_val = memory_history + memory_instant
    
    # --- 3. Noise & Field ---
    xi_t = noise_traj[step_idx - 1] 
    
    # Coupling along Z
    eff_field_contrib = xi_t + memory_val
    if coupling_type == "z":
         eff_field_contrib = jnp.array([0.0, 0.0, eff_field_contrib[2]])

    effective_field = B_field + eff_field_contrib
    
    # --- 4. Integration ---
    dSdt = jnp.cross(S_t, effective_field)
    S_next = S_t + dSdt * dt
    
    # --- 5. Normalization (Preserve TWA Length sqrt(3)) ---
    target_norm = jnp.linalg.norm(S_t)
    S_next_renorm = S_next / (jnp.linalg.norm(S_next) + 1e-12) * target_norm
    
    new_state = state.at[step_idx].set(S_next_renorm)
    return new_state, new_state[step_idx]

def run_twa_bundle(keys, t_grid, eta, omega_c, s, kBT, B_field, g, initial_direction, width_scale, batch_size=100):
    """
    Runs TWA with:
    1. Substeps (Fine grid) to capture Super-Ohmic physics.
    2. Batching to prevent Memory Errors (OOM).
    3. Progress Bar to track execution.
    """
    # 1. Setup Fine Grid
    dt= t_grid[1] - t_grid[0]
    
    num_steps_coarse = t_grid.shape[0]

    # 2. Pre-compute Kernel on FINE grid (Shared by all batches)
    gamma_kernel_fine = compute_memory_kernel(t_grid, eta, omega_c, s, g)
    
    # 3. Define the Batch Processor (JIT compiled for speed)
    @jax.jit
    def process_batch(batch_keys):
        # A. Key Splitting
        split_keys = jax.vmap(jax.random.split)(batch_keys)
        sampling_keys = split_keys[:, 0, :]
        noise_keys = split_keys[:, 1, :]
        
        # B. Sample Initial Conditions
        s_inits = jax.vmap(discrete_spin_sampling_factorized, in_axes=(0, None, None))(
            sampling_keys, initial_direction, width_scale
        )
        
        # C. Generate Noise on FINE grid
        noises_fine = jax.vmap(generate_noise, in_axes=(0, None, None, None, None, None, None))(
            noise_keys, t_grid, eta, omega_c, s, kBT, g
        )
        
        # D. Solver Logic
        def solve_single_trajectory(s0, single_noise):
            num_fine = single_noise.shape[0]
            # Initialize history array for the FINE grid
            history_init = jnp.zeros((num_fine, 3)).at[0].set(s0)
            
            def scan_body(carry, idx):
                # Use the fine dt and fine kernel
                return n_markovian_step(carry, idx, single_noise, gamma_kernel_fine, B_field, dt)

            # Run scan
            final_state_array, _ = jax.lax.scan(scan_body, history_init, jnp.arange(1, num_fine))
            
            # Downsample to match coarse grid
            return final_state_array

        # E. Run Solver for Batch
        batch_trajectories = jax.vmap(solve_single_trajectory)(s_inits, noises_fine)
        
        # F. Return Sum (we average later to save memory)
        return jnp.sum(batch_trajectories, axis=0)

    # 4. Batch Loop with Progress Bar
    n_trajs = keys.shape[0]
    total_sum = jnp.zeros((num_steps_coarse, 3))
    
    # Calculate number of batches for the progress bar
    # We iterate over the starting index 'i'
    # tqdm wrapper goes here:
    for i in tqdm(range(0, n_trajs, batch_size), desc="Simulating TWA Batches"):
        # Slice keys for current batch
        # Handle the last batch (which might be smaller than batch_size)
        batch_keys = keys[i : i + batch_size]
        
        # Run batch (JIT compiled)
        batch_sum = process_batch(batch_keys)
        
        # Accumulate
        total_sum = total_sum + batch_sum
        
    # 5. Final Average
    return total_sum / n_trajs