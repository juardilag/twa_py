import jax
import jax.numpy as jnp
from functions import make_tavis_cummings_system_functions
import time
import diffrax
import pandas as pd

def run_benchmark(trajectory_counts, params):
    
    # Define Devices
    try:
        cpu = jax.devices("cpu")[0]
        gpus = jax.devices("gpu")
        gpu = gpus[0] if len(gpus) > 0 else None
    except:
        gpu = None
        print("No GPU found. Benchmarking CPU only.")

    # Prepare Solver components once
    drift, diffusion = make_tavis_cummings_system_functions(
        params['omega_c'], params['omega_s'], params['g'], 
        params['kappa'], params['gamma_d'], params['gamma_u'], params['N']
    )
    t_eval = jnp.linspace(0, params['t_max'], int(1e3))

    # Single Trajectory
    @jax.jit
    def solve_one(y0, key):
        bm = diffrax.VirtualBrownianTree(t_eval[0], t_eval[-1]+0.1, tol=1e-3, shape=(2+4*params['N'],), key=key)
        term = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, bm))
        sol = diffrax.diffeqsolve(term, diffrax.Heun(), t0=t_eval[0], t1=t_eval[-1], dt0=0.01, y0=y0, saveat=diffrax.SaveAt(ts=t_eval))
        return sol.ys
    
    # Vectorized Batch of Trajectories
    @jax.jit
    def solve_batch(y0_batch, keys):
        return jax.vmap(solve_one)(y0_batch, keys)
    
    # The generation of initial conditions can also be made in CPU or GPU
    def get_data(n, device):
        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Sampling
        a = 0.5 * jax.random.normal(k1, (n, 1)) + 1j * 0.5 * jax.random.normal(k2, (n, 1))
        sx = 2.0 * jax.random.bernoulli(k3, 0.5, (n, params['N'])) - 1.0
        sy = 2.0 * jax.random.bernoulli(k4, 0.5, (n, params['N'])) - 1.0
        sz = jnp.ones((n, params['N']))
        y0 = jnp.concatenate([jnp.real(a), jnp.imag(a), jnp.stack([sx,sy,sz], axis=2).reshape(n, 3*params['N'])], axis=1)
        
        keys = jax.random.split(key, n)
        
        # Move to device
        return jax.device_put(y0, device), jax.device_put(keys, device)
    
    results = []

    for n in trajectory_counts:
        print(f"\n--- Benchmarking N_traj = {n} ---")

        # Single Core JIT Execution
        # Only for small N because it's terribly slow
        if n <= 1000: 
            y0, keys = get_data(n, cpu)
            # Warmup
            _ = solve_one(y0[0], keys[0]) 
            
            start = time.time()
            # The "For Loop" approach
            for i in range(n):
                _ = solve_one(y0[i], keys[i])
            end = time.time()
            results.append({"n_traj": n, "device": "Naive Loop (CPU)", "time": end - start})
            print(f"Naive Loop: {end-start:.4f} s")
        else:
            # Estimate linear scaling for plotting
            results.append({"n_traj": n, "device": "Naive Loop (CPU)", "time": None})

        # Multicore CPU JIT Execution
        if n <= 10000:
            y0, keys = get_data(n, cpu)
            _ = solve_batch(y0, keys).block_until_ready() # Warmup 

            start = time.time()
            _ = solve_batch(y0, keys).block_until_ready()
            end = time.time()
            results.append({"n_traj": n, "device": "Vectorized CPU", "time": end - start})
            print(f"Vectorized CPU: {end-start:.4f} s")
        else:
            # Estimate linear scaling for plotting
            results.append({"n_traj": n, "device": "Naive Loop (CPU)", "time": None})

        # Vectorized GPU JIT Execution
        y0, keys = get_data(n, gpu)
        _ = solve_batch(y0, keys).block_until_ready() # Warmup
        
        start = time.time()
        _ = solve_batch(y0, keys).block_until_ready()
        end = time.time()
        results.append({"n_traj": n, "device": "Vectorized GPU", "time": end - start})
        print(f"Vectorized GPU: {end-start:.4f} s")

    return pd.DataFrame(results)




