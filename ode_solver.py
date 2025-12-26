import jax
import jax.numpy as jnp
from spins import make_system_functions
from initial_samplings import discrete_spin_sampling
import diffrax
from tqdm import tqdm


def run_single_batch(s_batch, t_eval, args, key, term, solver):
    def solve_single(s_init):
        return diffrax.diffeqsolve(
            term, solver, t0=t_eval[0], t1=t_eval[-1], dt0=0.005,
            y0=s_init, args=args, saveat=diffrax.SaveAt(ts=t_eval),
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=40000
        ).ys
    
    # Run simulation for the batch
    batch_traj = jax.vmap(solve_single)(s_batch)
    
    # Calculate statistics inside JIT to avoid transferring big data
    # Sum over the batch dimension (axis 0)
    batch_sum = jnp.sum(batch_traj, axis=0)
    batch_sq_sum = jnp.sum(batch_traj**2, axis=0)
    
    return batch_sum, batch_sq_sum


def solve_twa_batched_stats(key, n_total, batch_size, t_eval, hamiltonian, jump_ops, args):
    """
    Runs TWA in batches and returns only Mean and Variance to save memory.
    """
    # Setup System
    drift, diffusion = make_system_functions(hamiltonian, jump_ops)
    n_noise_channels = 2 * len(jump_ops)
    
    # Setup Solver
    if n_noise_channels > 0:
        bm = diffrax.VirtualBrownianTree(t_eval[0], t_eval[-1], tol=1e-3, shape=(n_noise_channels,), key=key)
        term = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, bm))
        solver = diffrax.Heun()
    else:
        term = diffrax.ODETerm(drift)
        solver = diffrax.Dopri5()

    # Accumulators
    total_sum = jnp.zeros((len(t_eval), 3))
    total_sq_sum = jnp.zeros((len(t_eval), 3))
    
    num_batches = int(jnp.ceil(n_total / batch_size))
    
    print(f"Starting Simulation: {n_total} trajectories in {num_batches} batches...")
    
    for i in tqdm(range(num_batches)):
        # 1. Manage Random Keys for this batch
        iter_key, key = jax.random.split(key)
        batch_key, sample_key = jax.random.split(iter_key)
        
        # 2. Determine batch size (handle last batch)
        current_batch_size = min(batch_size, n_total - i*batch_size)
        
        # 3. Sample Initial Conditions
        # (Assuming discrete sampling function is available)
        s0_batch = discrete_spin_sampling(sample_key, current_batch_size, initial_z=-1.0)
        
        # 4. Run Batch on GPU
        # Note: We pass the pre-compiled term/solver to avoid re-compiling
        b_sum, b_sq_sum = run_single_batch(s0_batch, t_eval, args, batch_key, term, solver)
        
        # 5. Accumulate results on CPU (or GPU buffer)
        total_sum += b_sum
        total_sq_sum += b_sq_sum
        
    # Final Statistics
    mean_traj = total_sum / n_total
    # Var = E[x^2] - (E[x])^2
    var_traj = (total_sq_sum / n_total) - mean_traj**2
    
    return mean_traj, var_traj


