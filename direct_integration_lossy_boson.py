import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized, sample_coherent_discrete_rings
from lossy_boson import solve_dynamics_vacuum, get_initial_state
from tqdm.auto import tqdm

@jax.jit
def generate_complete_noise(key, t_grid, omega_0, kappa, g, n_photons_initial=0.0):
    dt = t_grid[1] - t_grid[0]
    # Necesitamos 4 llaves: 2 para el estado inicial de la cavidad, 2 para el baño
    k_init_re, k_init_im, k_re, k_im = jax.random.split(key, 4)
    
    # 1. Ruido Transitorio (ESTADO COHERENTE CORRECTO)
    # alpha_0 = Campo Medio + Fluctuaciones del Vacío (Wigner)
    mean_field = jnp.sqrt(n_photons_initial) # Asume fase 0 (campo coherente real)
    vacuum_fluc_re = jax.random.normal(k_init_re) * jnp.sqrt(0.5)
    vacuum_fluc_im = jax.random.normal(k_init_im) * jnp.sqrt(0.5)
    
    # El estado inicial es una Gaussiana centrada en el campo medio
    alpha_0 = mean_field + (vacuum_fluc_re + 1j * vacuum_fluc_im)
    transient_alpha = alpha_0 * jnp.exp(-(1j * omega_0 + 0.5 * kappa) * t_grid)
    
    # 2. Ruido Estacionario del Baño (Wigner Vacuum)
    dw_re = jax.random.normal(k_re, shape=t_grid.shape) * jnp.sqrt(0.5 * dt)
    dw_im = jax.random.normal(k_im, shape=t_grid.shape) * jnp.sqrt(0.5 * dt)
    white_noise = dw_re + 1j * dw_im
    
    # EOM de la cavidad para el baño
    def cavity_step(alpha_prev, dW):
        d_alpha = -(1j * omega_0 + 0.5 * kappa) * alpha_prev * dt + jnp.sqrt(0.5 * kappa) * dW
        return alpha_prev + d_alpha, alpha_prev + d_alpha

    _, bath_alpha = jax.lax.scan(cavity_step, 0j, white_noise)
    
    # 3. Campo Total de la Cavidad
    alpha_total = transient_alpha + bath_alpha
    phi_noise = 2.0 * jnp.real(alpha_total)
    
    # Xi = 2 * g * phi (Fuerza sobre el espín)
    xi_x = 2.0 * g * phi_noise
    return jnp.zeros((t_grid.shape[0], 3)).at[:, 0].set(xi_x)


def compute_memory_kernel(tau_grid, omega_0, kappa, g=1.0):
    """
    Theoretical Memory Kernel: gamma^R(t) magnitude.
    Captures the 4 * g^2 * exp(-kappa*t/2) * sin(w0*t) structure.
    """
    gamma_kernel = 4.0 * jnp.exp(-0.5 * kappa * tau_grid) * jnp.sin(omega_0 * tau_grid)
    return (g**2) * gamma_kernel

@jax.jit
def compute_effective_field(S_state, history_array, step_idx, gamma_kernel, 
                           noise_traj, B_field, dt):
    """
    Campo efectivo calibrado para DTWA.
    Factores: Ext=1.0, Noise=0.5, Memory=0.5
    """
    N = history_array.shape[0]
    indices = jnp.arange(N)
    lag_indices = step_idx - indices
    gamma_causal = jnp.where(lag_indices > 0, 
                             jnp.take(gamma_kernel, lag_indices, mode='fill', fill_value=0.0), 
                             0.0)

    # Memoria a 0.5: Compensa el 2.0 del integrador y mapea el kernel 4g^2 a la fuerza g.
    memory_x = 0.5*(jnp.dot(gamma_causal, history_array[:, 0]) * dt)
    
    # Ruido a 0.5: Recupera la escala física.
    xi_t = noise_traj[jnp.clip(step_idx, 0, noise_traj.shape[0]-1)]
    xi_field = 0.5*xi_t[0]
    
    # Campo Externo a 1.0 para mantener las frecuencias en resonancia
    eff_field_x = 0.5*B_field[0] + xi_field - memory_x
    
    return jnp.array([eff_field_x, 0.5*B_field[1], 0.5*B_field[2]])

@jax.jit
def heun_step_non_markovian(state_trajectory, step_idx, noise_traj, gamma_kernel, B_field, dt):
    curr_idx = step_idx - 1
    S_curr = state_trajectory[curr_idx]
    
    def get_rotation_params(S, idx, traj):
        B_eff = compute_effective_field(S, traj, idx, gamma_kernel, noise_traj, B_field, dt)
        
        # The 2.0 matching Lindblad/Pauli algebra
        b_mag = jnp.linalg.norm(B_eff) + 1e-16
        omega = 2.0 * b_mag 
        axis = B_eff / b_mag # Strictly unitary axis prevents explosions
        
        return axis, omega * dt

    # --- PREDICTOR ---
    axis_p, angle_p = get_rotation_params(S_curr, curr_idx, state_trajectory)
    S_pred = (S_curr * jnp.cos(angle_p) + 
              jnp.cross(axis_p, S_curr) * jnp.sin(angle_p) + 
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(angle_p)))
    
    # --- CORRECTOR ---
    traj_with_pred = state_trajectory.at[step_idx].set(S_pred)
    axis_c, angle_c = get_rotation_params(S_pred, step_idx, traj_with_pred)
    
    angle_avg = 0.5 * (angle_p + angle_c)
    
    S_next = (S_curr * jnp.cos(angle_avg) + 
              jnp.cross(axis_p, S_curr) * jnp.sin(angle_avg) + 
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(angle_avg)))
    
    return state_trajectory.at[step_idx].set(S_next), S_next


def run_twa_bundle(keys, t_grid, omega_0, kappa, B_field, g, n_photons_initial, initial_direction, coupling_type, batch_size=1000):
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    gamma_kernel_fine = compute_memory_kernel(t_grid, omega_0, kappa, g)
    
    def solve_single_trajectory(key):
        # We only need 2 splits now because generate_complete_noise does its own bath split internally
        k_samp, k_noise = jax.random.split(key)
        
        s0 = discrete_spin_sampling_factorized(k_samp, initial_direction)
        
        # Generates BOTH transient and bath noise
        noise_traj = generate_complete_noise(k_noise, t_grid, omega_0, kappa, g, n_photons_initial)
        
        history_init = jnp.zeros((num_steps, 3)).at[0].set(s0)
        
        def scan_body(carry, idx):
            return heun_step_non_markovian(carry, idx, noise_traj, gamma_kernel_fine, B_field, dt)

        final_traj, _ = jax.lax.scan(scan_body, history_init, jnp.arange(1, num_steps))
        return final_traj

    @jax.jit
    def process_batch_sum(batch_keys):
        batch_trajs = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_trajs, axis=0)

    total_sum = jnp.zeros((num_steps, 3))
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"Starting DTWA: {n_total} trajectories in {n_batches} batches.")
    
    for i in tqdm(range(n_batches), desc="DTWA Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        total_sum = total_sum + process_batch_sum(current_keys)
        
    return total_sum / n_total


def run_normalized_simulation(g_ratio, kappa_ratio, B_field_unit, v_init, tau_max, omega_0, n_photons_initial, num_steps, N=50, coupling = 'full'):
    """
    Runs both QuTiP and TWA simulations using normalized parameters.
    
    Returns:
        dict: Contains t_grid, QuTiP expectation values, and TWA results.
    """
    # 1. Scale parameters relative to omega_0
    kappa = kappa_ratio * omega_0
    g = g_ratio * omega_0
    # B_field input is treated as the relative magnitude/direction
    B_scaled = jnp.array(B_field_unit) * omega_0 
    
    # 2. Setup Time Grid (Dimensionless Scaling)
    t_max = tau_max / omega_0
    t_grid = jnp.linspace(0, t_max, num_steps)
    
    # 3. Initial State
    rho0 = get_initial_state(v_init, n_photons_initial, N)
    
    # 4. Run QuTiP Solver
    res = solve_dynamics_vacuum(
        Bx=B_scaled[0], 
        By=B_scaled[1], 
        Bz=B_scaled[2], 
        wa=omega_0, 
        g=g, 
        kappa=kappa, 
        times=t_grid, 
        rho0=rho0, 
        N=N
    )
    
    # 5. Run TWA Simulation
    n_trajectories = 50_000 
    master_key = jax.random.PRNGKey(42)
    keys = jax.random.split(master_key, n_trajectories)

    twa_results_raw = run_twa_bundle(
        keys=keys, 
        t_grid=t_grid, 
        omega_0=omega_0, 
        kappa=kappa, 
        B_field=B_scaled, 
        n_photons_initial = n_photons_initial,
        g=g, 
        initial_direction=jnp.array(v_init),
        coupling_type=coupling,
        batch_size=10_000
    )

    # 6. Organize and Return Data
    return {
        "t_grid": t_grid,
        "omega_0": omega_0,
        "qutip": {
            "expect_x": res.expect[0],
            "expect_y": res.expect[1],
            "expect_z": res.expect[2],
            "boson_num": res.expect[3]
        },
        "twa": {
            "expect_x": twa_results_raw[:, 0],
            "expect_y": twa_results_raw[:, 1],
            "expect_z": twa_results_raw[:, 2]
        }
    }