import jax.numpy as jnp
from jax import vmap
import jax
from initial_samplings import discrete_spin_sampling_factorized
from tqdm.auto import tqdm

jax.config.update("jax_enable_x64", True)

def get_spectral_density(omega, eta, omega_c, s):
    safe_omega = jnp.where(omega > 0, omega, 1.0) 
    J_val = eta * safe_omega * jnp.power(safe_omega / omega_c, s - 1) * jnp.exp(-safe_omega / omega_c)
    return jnp.where(omega > 0, J_val, 0.0)


# Memory Kernel
def compute_memory_kernel(tau_grid, eta, omega_c, s, g=1.0, num_omega=10_000):
    omega_max = 50 * omega_c
    
    omega_grid = jnp.linspace(1e-1, omega_max, num_omega)
    A_vals = 2.0 * get_spectral_density(omega_grid, eta, omega_c, s)
    sin_matrix = jnp.sin(jnp.outer(tau_grid, omega_grid))

    integrand = (A_vals[None, :] / jnp.pi) * sin_matrix
    integral = jnp.trapezoid(integrand, x=omega_grid, axis=1)
    
    return g**2 * integral

# Noise Generation
def setup_noise_parameters(t_grid, eta, omega_c, s, kBT, g=1.0, num_omega=2000):
    omega_max = 50.0 * omega_c
    omega_grid = jnp.linspace(1e-1, omega_max, num_omega)
    d_omega = omega_grid[1] - omega_grid[0]
    
    # 1. Spectral Amplitude
    safe_omega = jnp.maximum(omega_grid, 1e-12)
    J_omega = eta * safe_omega * jnp.power(safe_omega / omega_c, s - 1) * jnp.exp(-safe_omega / omega_c)
    
    x = safe_omega / (2.0 * kBT)
    thermal_factor = 1.0 / jnp.tanh(x)
    
    # Amplitude coefficient: g * sqrt( J(w) * coth(...) * dw / pi )
    # variance_density = (2*J * coth) / (2pi) = J*coth/pi
    amplitude = g * jnp.sqrt((J_omega * thermal_factor * d_omega) / jnp.pi)
    
    # 2. Construct the Evolution Operator Matrix (Time x Freq)
    # We use the identity: A*cos(wt) + B*sin(wt) = Re[ (A + iB) * e^(-iwt) ]
    time_evolution = jnp.exp(-1j * jnp.outer(t_grid, omega_grid))
    transfer_matrix = time_evolution * amplitude[None, :]
    
    return transfer_matrix


@jax.jit
def generate_noise_fast(key, transfer_matrix):
    num_omega = transfer_matrix.shape[1]
    
    # Generate Complex Normal Noise: Z = A + iB
    # A, B ~ N(0, 1)
    # Standard complex normal has var 1 (0.5 real, 0.5 imag), 
    # but we want A~N(0,1), B~N(0,1), so we generate standard normal complex 
    key_a, key_b = jax.random.split(key)
    
    noise_A = jax.random.normal(key_a, (num_omega, 3))
    noise_B = jax.random.normal(key_b, (num_omega, 3))
    
    Z_stoch = noise_A + 1j * noise_B

    complex_trajectory = transfer_matrix @ Z_stoch
    return jnp.real(complex_trajectory)


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
    # Total variance = Integral[ S_xi(w) * dw/pi ].
    variance_density = (A_omega * thermal_factor) / (2.0 * jnp.pi)
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


def compute_effective_field(S_state, history_array, step_idx, gamma_kernel, 
                            noise_traj, B_field, dt, coupling_type="z"):
    # 1. Vectorized Memory Convolution
    N = history_array.shape[0]
    indices = jnp.arange(N)
    
    # Calculate the lag indices: k = step_idx - j
    # valid lags are k > 0. 
    # k <= 0 implies current time or future (which should be 0)
    lag_indices = step_idx - indices

    # Fetch Gamma values. 
    # mode='fill' ensures that any index < 0 (future) or > len (too old) becomes 0.0
    # This automatically handles the "causality" mask for the future.
    gamma_vals = jnp.take(gamma_kernel, lag_indices, mode='fill', fill_value=0.0)
    
    # Explicitly zero out the current time step (lag=0) because it is handled 
    # separately as 'memory_instant'.
    # We use 'where' on the 1D array (fast) rather than the 2D array.
    gamma_causal = jnp.where(lag_indices > 0, gamma_vals, 0.0)

    # Fast Contraction: Vector (N) @ Matrix (N, 3) -> Vector (3)
    memory_history = jnp.dot(gamma_causal, history_array) * dt
    memory_instant = gamma_kernel[0] * S_state * dt
    memory_val = memory_history + memory_instant

    # 3. Noise
    # Safe clipping to ensure we don't crash if index is slightly off
    xi_t = noise_traj[jnp.clip(step_idx, 0, noise_traj.shape[0]-1)]
    eff_field_contrib = xi_t + memory_val
    
    if coupling_type == "z":
         eff_field_contrib = jnp.array([0.0, 0.0, eff_field_contrib[2]])

    return B_field + eff_field_contrib


# --- 2. The Heun Integrator ---
@jax.jit
def heun_step_non_markovian(state_trajectory, step_idx, noise_traj, gamma_kernel, B_field, dt):
    # Current time index (n)
    curr_idx = step_idx - 1
    S_curr = state_trajectory[curr_idx]
    
    # --- STAGE 1: PREDICTOR (Euler) ---
    # Calculate Field at t_n
    B_eff_curr = compute_effective_field(
        S_curr, state_trajectory, curr_idx, 
        gamma_kernel, noise_traj, B_field, dt
    )
    
    dS_1 = jnp.cross(S_curr, B_eff_curr)
    S_pred = S_curr + dS_1 * dt
    
    # --- STAGE 2: CORRECTOR ---
    # Temporarily update history to allow convolution to "see" the future prediction
    traj_with_pred = state_trajectory.at[step_idx].set(S_pred)
    
    # Calculate Field at t_{n+1}
    B_eff_next = compute_effective_field(
        S_pred, traj_with_pred, step_idx, 
        gamma_kernel, noise_traj, B_field, dt
    )
    
    dS_2 = jnp.cross(S_pred, B_eff_next)
    
    # Heun Update: Average the slopes
    S_next = S_curr + 0.5 * (dS_1 + dS_2) * dt
    
    # --- STAGE 3: NORMALIZATION ---
    # Crucial for stability in long simulations
    target_norm = jnp.linalg.norm(S_curr)
    S_next_renorm = S_next / (jnp.linalg.norm(S_next) + 1e-12) * target_norm
    
    # Write final result
    new_state_traj = state_trajectory.at[step_idx].set(S_next_renorm)
    
    return new_state_traj, S_next_renorm


def run_twa_bundle(keys, t_grid, eta, omega_c, s, kBT, B_field, g, initial_direction, batch_size=2000):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    # 1. Pre-compute Kernel (Analytical is instant)
    print("1. Computing Memory Kernel...")
    # Make sure to use the Analytical function from previous turn for speed
    gamma_kernel_fine = compute_memory_kernel(t_grid, eta, omega_c, s, g)
    
    # 2. Pre-compute Noise Matrix (Shared by all)
    print("2. Pre-computing Noise Transfer Matrix...")
    noise_transfer_matrix = setup_noise_parameters(t_grid, eta, omega_c, s, kBT, g)
    
    # 3. Define the Single Trajectory Solver
    # This generates noise internally to save VRAM
    def solve_single_trajectory(key):
        k_samp, k_noise = jax.random.split(key)
        
        # A. Sample Initial State
        s0 = discrete_spin_sampling_factorized(k_samp, initial_direction)
        
        # B. Generate Noise (Fast Matrix Multiply)
        # Note: We generate this JUST for this one trajectory
        noise_traj = generate_noise_fast(k_noise, noise_transfer_matrix)
        
        # C. Initialize History
        history_init = jnp.zeros((num_steps, 3)).at[0].set(s0)
        
        # D. Time Loop (Scan)
        def scan_body(carry, idx):
            # carry is the state_trajectory
            return heun_step_non_markovian(carry, idx, noise_traj, gamma_kernel_fine, B_field, dt)

        final_traj, _ = jax.lax.scan(scan_body, history_init, jnp.arange(1, num_steps))
        return final_traj

    # 4. Batch Processor
    # We map the solver over the batch keys and sum immediately
    @jax.jit
    def process_batch_sum(batch_keys):
        batch_trajs = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_trajs, axis=0)

    # 5. Main Loop
    total_sum = jnp.zeros((num_steps, 3))
    
    # Calculate batches
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"3. Starting Simulation: {n_total} trajectories in {n_batches} batches.")
    
    for i in tqdm(range(n_batches), desc="TWA Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        
        current_keys = keys[start_idx:end_idx]
        
        # Run optimized batch
        batch_partial_sum = process_batch_sum(current_keys)
        
        # Accumulate
        total_sum = total_sum + batch_partial_sum
        
    # 6. Final Average
    return total_sum / n_total


def compute_exact_expectation_value(t_grid, initial_state, eta, omega_c, s, kBT, g, num_omega=10000):
    """
    Calculates the Exact Quantum Mechanical Expectation Value <Sx(t)>
    for the Pure Dephasing Spin-Boson model (B=0).
    
    Returns:
        sx_t, sy_t, sz_t: Arrays of expectation values.
    """
    # 1. Normalize Initial State
    S0 = jnp.array(initial_state)
    norm = jnp.linalg.norm(S0)
    sx0, sy0, sz0 = S0 / norm
    
    # 2. Frequency Grid
    omega = jnp.linspace(1e-5, 50.0 * omega_c, num_omega)
    
    # 3. Spectral Density A(w)
    A_vals = 2.0 * eta * omega * jnp.power(omega / omega_c, s - 1) * jnp.exp(-omega / omega_c)
    
    # 4. Decoherence Function Gamma(t)
    thermal_factor = 1.0 / jnp.tanh(omega / (2.0 * kBT))
    
    def get_gamma(t):
        integrand = (A_vals * thermal_factor / (2.0 * jnp.pi * omega**2)) * (1.0 - jnp.cos(omega * t))
        return g**2 * jnp.trapezoid(integrand, x=omega)
        
    gamma_t = jax.vmap(get_gamma)(t_grid)
    
    # 5. Phi(t)
    def get_phi(t):
        integrand = (A_vals / (jnp.pi * omega**2)) * (omega * t - jnp.sin(omega * t))
        return g**2 * jnp.trapezoid(integrand, x=omega)

    phi_t = jax.vmap(get_phi)(t_grid)
    
    # 6. Construct Expectation Values
    decay = jnp.exp(-gamma_t)
    sx_t = decay * (sx0 * jnp.cos(phi_t) + sy0 * jnp.sin(phi_t))
    sy_t = decay * (sy0 * jnp.cos(phi_t) - sx0 * jnp.sin(phi_t))
    sz_t = jnp.full_like(t_grid, sz0) # Sz is conserved
    
    return sx_t