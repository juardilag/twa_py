import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"    
import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized, sample_coherent_discrete_rings
from tqdm import tqdm

@jax.jit
def generate_markovian_noise(key, num_steps, dt, kappa):
    """
    Generates pure white noise increments for the explicit cavity integration.
    We no longer pre-integrate the transient response!
    """
    k_re, k_im = jax.random.split(key)
    # Exact variance for the OU decay over step dt
    exact_variance = 0.25 * (1.0 - jnp.exp(-kappa * dt))
    
    dw_re = jax.random.normal(k_re, shape=(num_steps,)) * jnp.sqrt(exact_variance)
    dw_im = jax.random.normal(k_im, shape=(num_steps,)) * jnp.sqrt(exact_variance)
    
    return dw_re + 1j * dw_im

@jax.jit
def heun_step_coupled_with_jumps(carry, step_idx, noise_traj, B_field, dt, g, omega_0, kappa, n_spins=1):
    # Unpack the 5-component state, statistical weight, and stochastic key
    S_curr, alpha_curr, weight, key = carry
    key, jump_key = jax.random.split(key)
    
    noise_inc = noise_traj[step_idx - 1]
    exact_decay = jnp.exp(-(1j * omega_0 + 0.5 * kappa) * dt)
    coupling_strength = g / jnp.sqrt(n_spins)

    # --- 1. PREDICTOR ---
    # Spin effective field depends on the CURRENT real part of the cavity field
    B_eff_p_x = 0.5 * B_field[0] - 2.0 * coupling_strength * jnp.real(alpha_curr)
    B_eff_p = jnp.array([B_eff_p_x, 0.5 * B_field[1], 0.5 * B_field[2]])
    
    b_mag_p = jnp.linalg.norm(B_eff_p) + 1e-16
    axis_p = B_eff_p / b_mag_p
    angle_p = 2.0 * b_mag_p * dt 
    
    S_pred = (S_curr * jnp.cos(angle_p) + 
              jnp.cross(axis_p, S_curr) * jnp.sin(angle_p) + 
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(angle_p)))
              
    # Cavity predictor (exact linear decay + Heun coupling push)
    alpha_pred = alpha_curr * exact_decay - 1j * coupling_strength * S_curr[0] * dt + noise_inc
    
    # --- 2. CORRECTOR ---
    B_eff_c_x = 0.5 * B_field[0] - 2.0 * coupling_strength * jnp.real(alpha_pred)
    B_eff_c = jnp.array([B_eff_c_x, 0.5 * B_field[1], 0.5 * B_field[2]])
    
    B_eff_avg = 0.5 * (B_eff_p + B_eff_c)
    b_mag_avg = jnp.linalg.norm(B_eff_avg) + 1e-16
    axis_avg = B_eff_avg / b_mag_avg
    angle_avg = 2.0 * b_mag_avg * dt
    
    S_next = (S_curr * jnp.cos(angle_avg) + 
              jnp.cross(axis_avg, S_curr) * jnp.sin(angle_avg) + 
              axis_avg * jnp.dot(axis_avg, S_curr) * (1.0 - jnp.cos(angle_avg)))
              
    alpha_next = alpha_curr * exact_decay - 1j * coupling_strength * 0.5 * (S_curr[0] + S_pred[0]) * dt + noise_inc

    # --- 3. THE QUANTUM JUMP (Third Derivative Operator) ---
    # Probability driven by the cubic interaction vertex prefactor
    P_jump = jnp.abs(coupling_strength / 4.0) * dt
    is_jump = jax.random.uniform(jump_key) < P_jump
    
    # Apply the discrete Wigner shift (\Delta = 0.5) to the cavity
    alpha_jumped = alpha_next + 0.5
    
    # Apply the corresponding macroscopic shift to the spin
    # NOTE: Adjust [0.0, 0.5, -0.5] algebraically based on your exact Schwinger map derivation
    delta_S = jnp.array([0.0, 0.5, -0.5]) 
    S_jumped = S_next + delta_S
    S_jumped = S_jumped / jnp.linalg.norm(S_jumped) * jnp.linalg.norm(S_next) # Conserve length
    
    # Apply the jump stochastically
    S_final = jnp.where(is_jump, S_jumped, S_next)
    alpha_final = jnp.where(is_jump, alpha_jumped, alpha_next)
    
    # Multiply trajectory weight by -1 on jump for destructive interference
    weight_final = jnp.where(is_jump, weight * -1.0, weight)
    
    new_carry = (S_final, alpha_final, weight_final, key)
    
    return new_carry, (S_final, alpha_final)

def run_twa_bundle(keys, t_grid, omega_0, kappa, B_field, g, n_photons_initial, initial_direction, batch_size=1000, n_spins=1):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    def solve_single_trajectory(key):
        k_samp_spin, k_samp_alpha, k_noise, k_jump = jax.random.split(key, 4)
        
        # 1. Sample Initial States
        s0 = discrete_spin_sampling_factorized(k_samp_spin, initial_direction, n_spins)
        alpha0 = sample_coherent_discrete_rings(k_samp_alpha, jnp.sqrt(n_photons_initial))
        
        # 2. Generate pure Markovian noise (no memory integration)
        noise_traj = generate_markovian_noise(k_noise, num_steps, dt, kappa)
        
        # 3. Setup Carry: (Spin, Cavity, Weight, Jump_Key)
        initial_weight = 1.0
        carry_init = (s0, alpha0, initial_weight, k_jump)
        
        def scan_body(carry, idx):
            return heun_step_coupled_with_jumps(
                carry, idx, noise_traj, B_field, dt, g, omega_0, kappa, n_spins
            )

        # 4. Scan explicit evolution
        (final_carry, (S_traj, alpha_traj)) = jax.lax.scan(scan_body, carry_init, jnp.arange(1, num_steps))
        
        # Extract the accumulated weight from the final carry
        final_weight = final_carry[2]
        
        # Apply the statistical weight to the observables!
        return jnp.vstack([s0, S_traj]) * final_weight, jnp.append(alpha0, alpha_traj) * final_weight

    @jax.jit
    def process_batch_sum(batch_keys):
        # We sum both the spin trajectories and the photon trajectories
        batch_S_trajs, batch_alpha_trajs = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_S_trajs, axis=0), jnp.sum(batch_alpha_trajs, axis=0)

    total_sum_S = jnp.zeros((num_steps, 3))
    total_sum_alpha = jnp.zeros(num_steps, dtype=jnp.complex64)
    
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"Starting Explicit Coupled DTWA: {n_total} trajectories in {n_batches} batches.")
    
    for i in tqdm(range(n_batches), desc="DTWA Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        batch_sum_S, batch_sum_alpha = process_batch_sum(current_keys)
        total_sum_S += batch_sum_S
        total_sum_alpha += batch_sum_alpha
        
    return total_sum_S / n_total, total_sum_alpha / n_total