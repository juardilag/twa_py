import jax
import jax.numpy as jnp
from functions import make_spin_system_functions, make_tavis_cummings_system_functions
from initial_samplings import discrete_spin_sampling, boson_sampling
import diffrax
from tqdm import tqdm

def solve_spin_twa(key, n_total, batch_size, t_eval, hamiltonian, jump_ops, args):
    """
    Solves open TWA in batches with independent noise for each trajectory.
    """
    # 1. Setup System Functions
    drift, diffusion = make_spin_system_functions(hamiltonian, jump_ops)
    n_noise_channels = 2 * len(jump_ops)
    
    # 2. Define the Batch Runner (Closure Style)
    # We define this *inside* to capture 'drift', 'diffusion', 't_eval', etc.
    # This avoids passing functions as arguments to JIT.
    
    @jax.jit
    def run_single_batch(s_batch, keys_batch):
        """
        s_batch: (Batch, 3) - Initial states
        keys_batch: (Batch,) - Unique PRNG keys for noise
        """
        # Function to solve ONE trajectory
        def solve_one(s_init, single_noise_key):
            if n_noise_channels > 0:
                bm = diffrax.VirtualBrownianTree(
                    t_eval[0], t_eval[-1], 
                    tol=1e-3, 
                    shape=(n_noise_channels,), 
                    key=single_noise_key  # Unique key for this particle
                )
                term = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, bm))
                solver = diffrax.Heun()
            else:
                term = diffrax.ODETerm(drift)
                solver = diffrax.Dopri5()
            
            # Solve
            sol = diffrax.diffeqsolve(
                term, solver, 
                t0=t_eval[0], t1=t_eval[-1], 
                dt0=0.005, 
                y0=s_init, 
                args=args, 
                saveat=diffrax.SaveAt(ts=t_eval),
                stepsize_controller=diffrax.ConstantStepSize(),
                max_steps=40000
            )
            return sol.ys

        # Vectorize over states AND keys
        batch_traj = jax.vmap(solve_one)(s_batch, keys_batch)
        
        # Calculate statistics immediately to save memory
        batch_sum = jnp.sum(batch_traj, axis=0)
        batch_sq_sum = jnp.sum(batch_traj**2, axis=0)
        
        return batch_sum, batch_sq_sum

    # 3. Execution Loop
    total_sum = jnp.zeros((len(t_eval), 3))
    total_sq_sum = jnp.zeros((len(t_eval), 3))
    
    num_batches = int(jnp.ceil(n_total / batch_size))
    print(f"Starting Simulation: {n_total} trajectories (Independent Noise)...")
    
    for i in tqdm(range(num_batches)):
        # Manage Random Keys
        iter_key, key = jax.random.split(key)
        
        # A. Split key for Sampling AND Noise
        sample_key, batch_noise_key = jax.random.split(iter_key)
        
        # Always simulate 'batch_size' trajectories
        current_s0 = discrete_spin_sampling(sample_key, batch_size, initial_z=1.0)
        current_keys = jax.random.split(batch_noise_key, batch_size)

        # 2. Run the fixed-size batch (Hit the cache every time)
        b_sum, b_sq_sum = run_single_batch(current_s0, current_keys)
        
        total_sum += b_sum
        total_sq_sum += b_sq_sum

    actual_simulated_count = num_batches * batch_size
    
    mean_traj = total_sum / actual_simulated_count
    var_traj = (total_sq_sum / actual_simulated_count) - mean_traj**2
    
    return mean_traj, var_traj



def solve_tavis_cummings_twa(
    key, 
    n_total, 
    batch_size, 
    t_eval, 
    omega_cavity, 
    omega_spin, 
    g, 
    kappa, 
    gamma_down, 
    gamma_up, 
    N,
    initial_spin_state="up" # 'up' (+1.0) or 'down' (-1.0)
):
    """
    Solves the Tavis-Cummings model using TWA with batched execution.
    Uses the user-provided 'discrete_spin_sampling' and 'boson_sampling'.
    
    Args:
        key: JAX PRNGKey
        n_total: Total number of trajectories to simulate
        batch_size: Number of trajectories per batch
        t_eval: Time points array
        ... (Physical parameters) ...
        initial_spin_state: "up" or "down" for the atoms
        
    Returns:
        results: Dictionary containing time-evolution of observables:
            - "photon_number": <a^dag a>
            - "sz_mean": <Sum_i s^z_i> / N
            - "sx_mean": <Sum_i s^x_i> / N
            - "sy_mean": <Sum_i s^y_i> / N
    """
    
    # 1. Setup System Functions (Drift & Diffusion)
    # Uses the 'make_tavis_cummings_system_functions' defined previously
    drift, diffusion = make_tavis_cummings_system_functions(
        omega_cavity, omega_spin, g, kappa, gamma_down, gamma_up, N
    )
    
    n_noise_channels = 2 + 4 * N
    args = None # Params captured by closure
    
    # -------------------------------------------------------------------------
    # 2. Batch Initialization using User Samplers
    # -------------------------------------------------------------------------
    def sample_initial_batch(rng_key, batch_n):
        k_boson, k_spin = jax.random.split(rng_key)
        
        # A. Photon Sampling (Vacuum |0>)
        # Returns (batch_n,) complex array
        a_complex = boson_sampling(k_boson, batch_n, initial_alpha=0.0)
        
        # Split into Real/Imag for the state vector
        a_re = jnp.real(a_complex)[:, None] # Shape (batch_n, 1)
        a_im = jnp.imag(a_complex)[:, None] # Shape (batch_n, 1)
        
        # B. Spin Sampling
        # We need N spins for EACH trajectory in the batch.
        # Total spins needed = batch_n * N
        total_spins = batch_n * N
        z_val = 1.0 if initial_spin_state == "up" else -1.0
        
        # Call user function: Returns (total_spins, 3)
        spins_bulk = discrete_spin_sampling(k_spin, n_trajectories=total_spins, initial_z=z_val)
        
        # Reshape to separate trajectories: (batch_n, N, 3)
        spins_structured = spins_bulk.reshape(batch_n, N, 3)
        
        # Flatten the (N, 3) part into (3N,) to match state vector format
        # [s1x, s1y, s1z, s2x, s2y, s2z, ...]
        spins_flat = spins_structured.reshape(batch_n, 3 * N)
        
        # Concatenate: [Re(a), Im(a), Spins...]
        # Output shape: (batch_n, 2 + 3N)
        y0 = jnp.concatenate([a_re, a_im, spins_flat], axis=1)
        
        return y0

    # -------------------------------------------------------------------------
    # 3. Batch Solver (JIT Compiled)
    # -------------------------------------------------------------------------
    @jax.jit
    def run_batch(y0_batch, noise_keys):
        
        def solve_one_traj(y0, noise_key):
            solver = diffrax.Heun() # Stratonovich solver required for TWA
            
            # Virtual Brownian Tree for noise
            bm = diffrax.VirtualBrownianTree(
                t_eval[0], t_eval[-1] + 0.1, 
                tol=1e-3, 
                shape=(n_noise_channels,), 
                key=noise_key
            )
            
            term = diffrax.MultiTerm(
                diffrax.ODETerm(drift), 
                diffrax.ControlTerm(diffusion, bm)
            )
            
            sol = diffrax.diffeqsolve(
                term, solver,
                t0=t_eval[0], t1=t_eval[-1],
                dt0=0.005,
                y0=y0,
                args=args,
                saveat=diffrax.SaveAt(ts=t_eval),
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3),
                max_steps=50000
            )
            return sol.ys # (Time, State_Dim)

        # Vectorize solving over the batch
        batch_ys = jax.vmap(solve_one_traj)(y0_batch, noise_keys)
        
        # --- Compute Observables ---
        
        # 1. Photon Number: <a^dag a>_sym = |a|^2
        a_re = batch_ys[:, :, 0]
        a_im = batch_ys[:, :, 1]
        photon_sq_mag = a_re**2 + a_im**2
        
        # 2. Collective Spins
        # Extract spins part (Batch, Time, 3N)
        spins_part = batch_ys[:, :, 2:]
        # Reshape to (Batch, Time, N, 3)
        spins_part = spins_part.reshape(batch_size, len(t_eval), N, 3)
        
        # Average over N atoms (Collective variables / N)
        # s_vector index 0=x, 1=y, 2=z
        sx_mean = jnp.mean(spins_part[:, :, :, 0], axis=2)
        sy_mean = jnp.mean(spins_part[:, :, :, 1], axis=2)
        sz_mean = jnp.mean(spins_part[:, :, :, 2], axis=2)
        
        # Sum over batch (to be averaged later)
        return {
            "sum_photon_sq": jnp.sum(photon_sq_mag, axis=0),
            "sum_sz": jnp.sum(sz_mean, axis=0),
            "sum_sx": jnp.sum(sx_mean, axis=0),
            "sum_sy": jnp.sum(sy_mean, axis=0)
        }

    # -------------------------------------------------------------------------
    # 4. Execution Loop
    # -------------------------------------------------------------------------
    acc_photon = jnp.zeros(len(t_eval))
    acc_sz = jnp.zeros(len(t_eval))
    acc_sx = jnp.zeros(len(t_eval))
    acc_sy = jnp.zeros(len(t_eval))
    
    num_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"Solving Tavis-Cummings TWA for N={N} atoms...")
    print(f"Total Trajectories: {n_total} | Batches: {num_batches}")
    
    for _ in tqdm(range(num_batches)):
        iter_key, key = jax.random.split(key)
        sample_key, noise_key = jax.random.split(iter_key)
        
        # 1. Sample Initial States (CPU or GPU)
        batch_y0 = sample_initial_batch(sample_key, batch_size)
        
        # 2. Generate Noise Keys
        batch_noise_keys = jax.random.split(noise_key, batch_size)
        
        # 3. Run Batch
        batch_res = run_batch(batch_y0, batch_noise_keys)
        
        # 4. Accumulate
        acc_photon += batch_res["sum_photon_sq"]
        acc_sz += batch_res["sum_sz"]
        acc_sx += batch_res["sum_sx"]
        acc_sy += batch_res["sum_sy"]

    # 5. Normalize Results
    actual_total = num_batches * batch_size
    
    # Photon: <n> = <|a|^2> - 1/2 (Wigner correction)
    final_photon = (acc_photon / actual_total) - 0.5
    
    final_sz = acc_sz / actual_total
    final_sx = acc_sx / actual_total
    final_sy = acc_sy / actual_total
    
    return {
        "photon_number": final_photon,
        "sz_mean": final_sz,
        "sx_mean": final_sx,
        "sy_mean": final_sy
    }


