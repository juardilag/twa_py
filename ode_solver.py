import jax
import jax.numpy as jnp
from spins import make_system_functions
from initial_samplings import discrete_spin_sampling
import diffrax
from tqdm import tqdm


def solve_twa_batched_stats(key, n_total, batch_size, t_eval, hamiltonian, jump_ops, args):
    """
    Solves open TWA in batches with independent noise for each trajectory.
    """
    # 1. Setup System Functions
    drift, diffusion = make_system_functions(hamiltonian, jump_ops)
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


