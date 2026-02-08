import jax.numpy as jnp
from jax import vmap
import jax
from initial_samplings import discrete_spin_sampling_single
from jax.scipy.special import gamma

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
    
    # --- Memory Calculation ---
    # We need Sum_{j=0}^{step-1} gamma(t_{step} - t_j) * S(t_j) * dt
    indices = jnp.arange(state.shape[0])
    
    # Mask for past (j < step_idx)
    past_mask = (indices < step_idx)
    
    # Kernel indices: we need gamma[step_idx - j]
    k_indices = (step_idx - indices)
    
    # Gather relevant kernel values (0 where mask is false)
    current_gamma = jnp.where(past_mask[:, None], gamma_kernel[k_indices][:, None], 0.0)
    
    # Convolve: Integral approx as sum * dt
    memory_val = jnp.sum(current_gamma * state, axis=0) * dt
    
    # --- Noise & Field ---
    xi_t = noise_traj[step_idx - 1] 
    
    # Selection of Coupling Axis
    if coupling_type == "z":
        # Noise/Memory along Z, but Torque = S x (B_z + ...)
        # Ensure we only take the Z-components of noise and memory
        eff_field_contrib = jnp.array([0.0, 0.0, xi_t[2] + memory_val[2]])
    else:
        eff_field_contrib = xi_t + memory_val

    effective_field = B_field + eff_field_contrib
    
    # --- Integration (Euler) ---
    dSdt = jnp.cross(S_t, effective_field)
    S_next = S_t + dSdt * dt
    
    # --- Normalization (CRITICAL FIX) ---
    # Instead of normalizing to 1.0, we normalize to the length of the previous state.
    # This preserves the TWA length (sqrt(3)) set by your sampler.
    target_norm = jnp.linalg.norm(S_t)
    S_next_renorm = S_next / (jnp.linalg.norm(S_next) + 1e-12) * target_norm
    
    new_state = state.at[step_idx].set(S_next_renorm)
    return new_state, new_state[step_idx]


def run_twa_bundle(keys, t_grid, eta, omega_c, s, kBT, B_field, g, initial_direction, width_scale):
    dt = t_grid[1] - t_grid[0]
    n_trajs = keys.shape[0]
    num_steps = t_grid.shape[0]
    
    # 1. Pre-compute Kernel
    tau_grid = jnp.linspace(0, t_grid[-1], num_steps)
    gamma_kernel = compute_memory_kernel(tau_grid, eta, omega_c, s, g)
    
    # 2. Key Management: Corrected Unpacking
    # Split each key and move the '2' to the front for unpacking
    split_keys = jax.vmap(jax.random.split)(keys)  # (N, 2, 2)
    sampling_keys = split_keys[:, 0, :]           # (N, 2)
    noise_keys = split_keys[:, 1, :]              # (N, 2)

    # 3. Vmap the sampler (now sampling_keys has the correct shape)
    s_inits = jax.vmap(discrete_spin_sampling_single, in_axes=(0, None, None))(
        sampling_keys, initial_direction, width_scale
    )
    
    # 4. Parallelize noise generation
    # Each trajectory now evolves under unique noise
    noises = jax.vmap(generate_noise, in_axes=(0, None, None, None, None, None, None))(
        noise_keys, t_grid, eta, omega_c, s, kBT, g
    )

    def solve_single_trajectory(s0, single_noise):
        num_steps = single_noise.shape[0]
        # Initialize history with the starting spin [cite: 42]
        history_init = jnp.zeros((num_steps, 3)).at[0].set(s0)
        
        # Define the scan body
        def scan_body(carry, idx):
            # n_markovian_step now returns (new_carry, step_output)
            return n_markovian_step(carry, idx, single_noise, gamma_kernel, B_field, dt)

        # Carry is the full 'state' array being updated
        # Trajectory is the 'S_next' gathered at each step
        final_state_array, trajectory = jax.lax.scan(scan_body, history_init, jnp.arange(1, num_steps))
        
        # We return the full array which includes the initial s0 at index 0
        return final_state_array

    # 5. Parallelize the ODE solver
    all_trajectories = jax.vmap(solve_single_trajectory)(s_inits, noises)
    
    # Average over trajectories
    return jnp.mean(all_trajectories, axis=0)