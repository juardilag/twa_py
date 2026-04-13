import os
os.environ["JAX_ENABLE_TRITON_GEMM"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["JAX_LOG_LEVEL"] = "error"    
import jax.numpy as jnp
import jax
from initial_samplings import discrete_spin_sampling_factorized
from tqdm import tqdm

@jax.jit(static_argnames=['num_steps'])
def generate_markovian_noise(key, num_steps, dt, kappa):
    """
    Generates pure white noise increments for the explicit cavity integration.
    """
    k_re, k_im = jax.random.split(key)
    # Exact variance for the OU decay over step dt
    exact_variance = 0.25 * (1.0 - jnp.exp(-kappa * dt))
    
    dw_re = jax.random.normal(k_re, shape=(num_steps,)) * jnp.sqrt(exact_variance)
    dw_im = jax.random.normal(k_im, shape=(num_steps,)) * jnp.sqrt(exact_variance)
    
    return dw_re + 1j * dw_im

@jax.jit
def heun_step_coupled_continuous(carry, step_idx, noise_traj, B_field, dt, g, omega_0, kappa, n_spins=1):
    """
    Standard Continuous TWA explicit step using Exponential Time Differencing.
    """
    S_curr, alpha_curr = carry
    
    noise_inc = noise_traj[step_idx - 1]
    
    # 1. Exact Linear Propagators (Exponential Time Differencing)
    z = 1j * omega_0 + 0.5 * kappa
    exact_decay = jnp.exp(-z * dt)
    
    # The EXACT analytical integral of the spin drive over the rotating timestep dt
    phi_drive = (1.0 - exact_decay) / z
    
    coupling_strength = g / jnp.sqrt(n_spins)

    # --- 1. PREDICTOR ---
    B_eff_p_x = 0.5 * B_field[0] + 2.0 * coupling_strength * jnp.real(alpha_curr)
    B_eff_p = jnp.array([B_eff_p_x, 0.5 * B_field[1], 0.5 * B_field[2]])
    
    b_mag_p = jnp.linalg.norm(B_eff_p) + 1e-16
    axis_p = B_eff_p / b_mag_p
    angle_p = 2.0 * b_mag_p * dt 
    
    S_pred = (S_curr * jnp.cos(angle_p) + 
              jnp.cross(axis_p, S_curr) * jnp.sin(angle_p) + 
              axis_p * jnp.dot(axis_p, S_curr) * (1.0 - jnp.cos(angle_p)))
              
    # Cavity predictor: We use phi_drive instead of dt!
    alpha_pred = alpha_curr * exact_decay - 1j * coupling_strength * S_curr[0] * phi_drive + noise_inc
    
    # --- 2. CORRECTOR ---
    B_eff_c_x = 0.5 * B_field[0] + 2.0 * coupling_strength * jnp.real(alpha_pred)
    B_eff_c = jnp.array([B_eff_c_x, 0.5 * B_field[1], 0.5 * B_field[2]])
    
    B_eff_avg = 0.5 * (B_eff_p + B_eff_c)
    b_mag_avg = jnp.linalg.norm(B_eff_avg) + 1e-16
    axis_avg = B_eff_avg / b_mag_avg
    angle_avg = 2.0 * b_mag_avg * dt
    
    S_next = (S_curr * jnp.cos(angle_avg) + 
              jnp.cross(axis_avg, S_curr) * jnp.sin(angle_avg) + 
              axis_avg * jnp.dot(axis_avg, S_curr) * (1.0 - jnp.cos(angle_avg)))
              
    # Cavity corrector: We use phi_drive instead of dt!
    alpha_next = alpha_curr * exact_decay - 1j * coupling_strength * 0.5 * (S_curr[0] + S_pred[0]) * phi_drive + noise_inc

    new_carry = (S_next, alpha_next)
    
    return new_carry, (S_next, alpha_next)

def run_coupled_twa_bundle(keys, t_grid, omega_0, kappa, B_field, g, n_photons_initial, initial_direction, batch_size=1000, n_spins=1):
    """
    Computes ONLY the 1D expectation values. Highly optimized for speed.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    def solve_single_trajectory(key):
        k_samp_spin, k_samp_alpha, k_noise = jax.random.split(key, 3)
        
        s0 = discrete_spin_sampling_factorized(k_samp_spin, initial_direction, n_spins)
        k_init_re, k_init_im = jax.random.split(k_samp_alpha)
        vacuum_fluc_re = jax.random.normal(k_init_re) * jnp.sqrt(0.5)
        vacuum_fluc_im = jax.random.normal(k_init_im) * jnp.sqrt(0.5)
        alpha0 = jnp.sqrt(n_photons_initial) + (vacuum_fluc_re + 1j * vacuum_fluc_im)
        
        noise_traj = generate_markovian_noise(k_noise, num_steps, dt, kappa)
        carry_init = (s0, alpha0)
        
        def scan_body(carry, idx):
            return heun_step_coupled_continuous(
                carry, idx, noise_traj, B_field, dt, g, omega_0, kappa, n_spins
            )

        _, (S_traj, alpha_traj) = jax.lax.scan(scan_body, carry_init, jnp.arange(1, num_steps))
        
        return jnp.vstack([s0, S_traj]), jnp.append(alpha0, alpha_traj)

    @jax.jit
    def process_batch_sum(batch_keys):
        batch_S, batch_alpha = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_S, axis=0), jnp.sum(batch_alpha, axis=0)

    total_sum_S = jnp.zeros((num_steps, 3))
    total_sum_alpha = jnp.zeros(num_steps, dtype=jnp.complex64)
    
    n_batches = int(jnp.ceil(n_total / batch_size))

    print(f"Starting Non-Integrated (Markovian) DTWA: {n_total} trajectories in {n_batches} batches.")
    
    for i in tqdm(range(n_batches), desc="DTWA Expectations"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        batch_sum_S, batch_sum_alpha = process_batch_sum(current_keys)
        total_sum_S += batch_sum_S
        total_sum_alpha += batch_sum_alpha
        
    return total_sum_S / n_total, total_sum_alpha / n_total

def run_time_integrated_light_correlation(keys, t_grid, omega_0, kappa, B_field, g, n_photons_initial, initial_direction, tau_steps=2000, batch_size=1000, n_spins=1):
    """
    Computes the 1D time-integrated correlation function C(tau) DIRECTLY in the time domain.
    Uses the Rolling Tangent Method to compute the exact unequal-time commutator for 
    interacting systems, replacing the empty-cavity approximation.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    # The number of points we can actually sum over for a given tau
    valid_length = num_steps - tau_steps
    tau_indices = jnp.arange(tau_steps)
    
    def solve_single_trajectory(key):
        k_samp_spin, k_samp_alpha, k_noise = jax.random.split(key, 3)
        
        # 1. INITIALIZATION
        s0 = discrete_spin_sampling_factorized(k_samp_spin, initial_direction, n_spins)
        k_init_re, k_init_im = jax.random.split(k_samp_alpha)
        vacuum_fluc_re = jax.random.normal(k_init_re) * jnp.sqrt(0.5)
        vacuum_fluc_im = jax.random.normal(k_init_im) * jnp.sqrt(0.5)
        
        alpha0_re = jnp.sqrt(n_photons_initial) + vacuum_fluc_re
        alpha0_im = vacuum_fluc_im
        alpha0_complex = alpha0_re + 1j * alpha0_im
        
        noise_traj = generate_markovian_noise(k_noise, num_steps, dt, kappa)
        
        # Base carry state split into strictly real components for jax.linearize
        carry_init_ri = (s0, alpha0_re, alpha0_im)
        
        # Initialize zero-buffers for the rolling tangents
        # Shape: (tau_steps, ...) for each variable in the carry state
        t_Re_init = jax.tree.map(lambda x: jnp.zeros((tau_steps,) + x.shape), carry_init_ri)
        t_Im_init = jax.tree.map(lambda x: jnp.zeros((tau_steps,) + x.shape), carry_init_ri)
        chi_acc_init = jnp.zeros(tau_steps, dtype=jnp.complex64)
        
        loop_state = (carry_init_ri, t_Re_init, t_Im_init, chi_acc_init)

        # 2. SCAN BODY WITH ROLLING TANGENTS
        def scan_body(loop_state, idx):
            carry_ri, t_Re, t_Im, chi_acc = loop_state
            
            # Wrapper to handle the Real/Imaginary split required by jax.linearize
            def pure_step_fn(c_ri):
                s_in, a_re_in, a_im_in = c_ri
                c_complex = (s_in, a_re_in + 1j * a_im_in)
                
                # 1. Capture the tuple structure expected by jax.lax.scan
                next_carry, step_output = heun_step_coupled_continuous(
                    c_complex, idx, noise_traj, B_field, dt, g, omega_0, kappa, n_spins
                )
                
                # 2. Extract the actual state variables from the carry
                s_next, a_next = next_carry
                
                # 3. Return the split real/imaginary state for the automatic differentiator
                return (s_next, jnp.real(a_next), jnp.imag(a_next))

            # Push forward the primary state and get the Jacobian-vector product function
            next_carry_ri, jvp_fn = jax.linearize(pure_step_fn, carry_ri)

            # Push all historical perturbations forward one time step
            next_t_Re = jax.vmap(jvp_fn)(t_Re)
            next_t_Im = jax.vmap(jvp_fn)(t_Im)

            # Shift buffers down to age them by 1 step
            shifted_t_Re = jax.tree.map(lambda x: jnp.roll(x, shift=1, axis=0), next_t_Re)
            shifted_t_Im = jax.tree.map(lambda x: jnp.roll(x, shift=1, axis=0), next_t_Im)

            # Inject the NEW perturbations at tau=0 (index 0)
            # updated_t_Re kicks alpha_re by 1.0
            updated_t_Re = (
                shifted_t_Re[0].at[0].set(0.0),  
                shifted_t_Re[1].at[0].set(1.0),  
                shifted_t_Re[2].at[0].set(0.0)   
            )
            
            # updated_t_Im kicks alpha_im by 1.0
            updated_t_Im = (
                shifted_t_Im[0].at[0].set(0.0),  
                shifted_t_Im[1].at[0].set(0.0),  
                shifted_t_Im[2].at[0].set(1.0)   
            )

            # Extract the alpha responses to calculate the commutator for this step
            # Note: The buffer at index `tau` represents the response to a kick `tau` steps ago
            resp_from_Re = updated_t_Re[1] + 1j * updated_t_Re[2]
            resp_from_Im = updated_t_Im[1] + 1j * updated_t_Im[2]
            
            # Form the exact Wigner commutator expectation value: 
            # 0.5 * (d_alpha / d_Re - i * d_alpha / d_Im)
            chi_step = 0.5 * (resp_from_Re - 1j * resp_from_Im)
            
            # Accumulate the time-integrated response
            new_chi_acc = chi_acc + chi_step * dt

            # Save the primal complex alpha for the classical Wigner correlation
            complex_alpha_out = next_carry_ri[1] + 1j * next_carry_ri[2]

            return (next_carry_ri, updated_t_Re, updated_t_Im, new_chi_acc), complex_alpha_out

        final_loop_state, alpha_traj = jax.lax.scan(scan_body, loop_state, jnp.arange(1, num_steps))
        
        # Extract the fully integrated exact commutator
        _, _, _, integrated_chi = final_loop_state
        
        # 3. EXACT TIME-DOMAIN SLIDING WINDOW (INTEGRATED)
        alpha_full = jnp.append(alpha0_complex, alpha_traj)
        
        def compute_integrated_C_tau(tau):
            # Grab the unshifted base array
            a_base = jax.lax.dynamic_slice_in_dim(alpha_full, 0, valid_length)
            # Grab the shifted array 
            a_shifted = jax.lax.dynamic_slice_in_dim(alpha_full, tau, valid_length)
            
            # Return the exact time-integral (sum * dt) for this specific tau
            return jnp.sum(jnp.conj(a_shifted) * a_base) * dt
        
        integrated_C_wigner = jax.vmap(compute_integrated_C_tau)(tau_indices)
        
        return integrated_C_wigner, integrated_chi

    @jax.jit
    def process_batch_sum(batch_keys):
        # We now return and sum BOTH the Wigner correlation and the dynamical Chi
        batch_C, batch_chi = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_C, axis=0), jnp.sum(batch_chi, axis=0)

    # Accumulators for the batch loop
    total_C_tau = jnp.zeros(tau_steps, dtype=jnp.complex64)
    total_chi_tau = jnp.zeros(tau_steps, dtype=jnp.complex64)
    
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"Computing Integrated Time-Domain Correlation & Linear Response up to tau={tau_steps} steps...")
    for i in tqdm(range(n_batches), desc="Autocorrelation Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        batch_C_sum, batch_chi_sum = process_batch_sum(current_keys)
        total_C_tau += batch_C_sum
        total_chi_tau += batch_chi_sum
        
    # Average over the total number of trajectories
    C_wigner_tau = total_C_tau / n_total
    chi_tau_exact = total_chi_tau / n_total
    
    # --- EXACT PHYSICAL CORRELATION ---
    # The 0.5 factor is already baked into the `chi_step` calculation inside the scan
    C_physical_tau = C_wigner_tau - chi_tau_exact
    
    tau_time = tau_indices * dt
    
    return tau_time, C_physical_tau

def run_time_integrated_spin_correlation(keys, t_grid, omega_0, kappa, B_field, g, n_photons_initial, initial_direction, tau_steps=2000, batch_size=1000, n_spins=1):
    """
    Computes the 1D time-averaged spin dipole correlation C(tau) = <S+(tau) S-(0)>.
    This is highly robust against Monte Carlo noise and requires ZERO vacuum subtraction.
    """
    dt = t_grid[1] - t_grid[0]
    num_steps = t_grid.shape[0]
    n_total = keys.shape[0]
    
    # SAFEGUARD: Ensure we never try to shift past the end of the array.
    max_allowed_tau = int(num_steps * 0.9)
    actual_tau_steps = min(tau_steps, max_allowed_tau)
    
    valid_length = num_steps - actual_tau_steps
    tau_indices = jnp.arange(actual_tau_steps)
    
    def solve_single_trajectory(key):
        k_samp_spin, k_samp_alpha, k_noise = jax.random.split(key, 3)
        
        s0 = discrete_spin_sampling_factorized(k_samp_spin, initial_direction, n_spins)
        k_init_re, k_init_im = jax.random.split(k_samp_alpha)
        vacuum_fluc_re = jax.random.normal(k_init_re) * jnp.sqrt(0.5)
        vacuum_fluc_im = jax.random.normal(k_init_im) * jnp.sqrt(0.5)
        alpha0 = jnp.sqrt(n_photons_initial) + (vacuum_fluc_re + 1j * vacuum_fluc_im)
        
        noise_traj = generate_markovian_noise(k_noise, num_steps, dt, kappa)
        carry_init = (s0, alpha0)
        
        def scan_body(carry, idx):
            return heun_step_coupled_continuous(
                carry, idx, noise_traj, B_field, dt, g, omega_0, kappa, n_spins
            )

        _, (S_traj, _) = jax.lax.scan(scan_body, carry_init, jnp.arange(1, num_steps))
        
        # S_full contains the spin trajectories. 
        S_full = jnp.vstack([s0, S_traj])
        
        # 1. Extract the macroscopic collective spin.
        # If your S_full shape is (time_steps, n_spins, 3), we sum over the spins.
        # If it is already (time_steps, 3), we just take the components.
        if S_full.ndim == 3:
            Sx = jnp.sum(S_full[:, :, 0], axis=1)
            Sy = jnp.sum(S_full[:, :, 1], axis=1)
        else:
            Sx = S_full[:, 0]
            Sy = S_full[:, 1]
            
        # 2. Construct the complex raising/lowering operators
        S_plus = Sx + 1j * Sy
        # In classical TWA, S_minus is simply the complex conjugate of S_plus
        S_minus = jnp.conj(S_plus)
        
        # --- THE EXACT TIME-DOMAIN SLIDING WINDOW ---
        def compute_C_tau(tau):
            # Base array evaluated at t = 0
            S_minus_base = jax.lax.dynamic_slice_in_dim(S_minus, 0, valid_length)
            
            # Shifted array evaluated at t = tau
            S_plus_shifted = jax.lax.dynamic_slice_in_dim(S_plus, tau, valid_length)
            
            # C(tau) = <S+(tau) S-(0)>
            return jnp.mean(S_plus_shifted * S_minus_base)
        
        # Vectorize over all requested delay times
        C_tau = jax.vmap(compute_C_tau)(tau_indices)
        return C_tau

    @jax.jit
    def process_batch_sum(batch_keys):
        batch_C = jax.vmap(solve_single_trajectory)(batch_keys)
        return jnp.sum(batch_C, axis=0)

    total_C_tau = jnp.zeros(actual_tau_steps, dtype=jnp.complex64)
    n_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"Computing Exact Spin Dipole Correlation up to tau={actual_tau_steps} steps...")
    for i in tqdm(range(n_batches), desc="Spin Autocorrelation"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        current_keys = keys[start_idx:end_idx]
        
        total_C_tau += process_batch_sum(current_keys)
        
    C_spin_tau = total_C_tau / n_total
    tau_time = tau_indices * dt
    
    # Notice we return this DIRECTLY. No vacuum subtraction needed!
    return tau_time, C_spin_tau

@jax.jit
def compute_spectrum(C_tau, tau_grid, omega_grid):
    """
    Computes the exact power spectrum using the one-sided Wiener-Khinchin theorem.
    No windowing is applied so the pure physics is preserved.
    """
    dt = tau_grid[1] - tau_grid[0]
    
    def compute_single_omega(omega):
        # 1. The correct physical phase for emission: -i * omega * tau
        integrand = C_tau * jnp.exp(-1j * omega * tau_grid)
        
        # 2. Trapezoidal rule correction: halve the tau=0 point.
        # This prevents overcounting the boundary which causes vertical offsets.
        integrand = integrand.at[0].multiply(0.5)
        
        # 3. Riemann sum
        one_sided_integral = jnp.sum(integrand) * dt
        
        # 4. Multiply by 2 and take the Real part to reconstruct the full symmetric integral
        return 2.0 * jnp.real(one_sided_integral)

    # Vectorize across your custom high-resolution frequency grid
    return jax.vmap(compute_single_omega)(omega_grid)