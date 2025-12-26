import jax
from spins import make_system_functions
import diffrax

def solve_spin_twa(s0_batch, t_eval, hamiltonian, jump_ops, args, key):
    """
    Solves TWA for ANY Hamiltonian and ANY Jump Operators.
    
    Args:
        s0_batch: Initial states (N_traj, 3)
        t_eval: Time array
        hamiltonian: Function H(s, args)
        jump_ops: List of Functions [L1(s, args), L2(s, args), ...]
        args: Parameter tuple passed to functions
        key: JAX PRNGKey
    """
    
    # 1. Build the Physics Engine
    drift, diffusion = make_system_functions(hamiltonian, jump_ops)
    
    # 2. Setup Brownian Motion
    # We need (2 * len(jump_ops)) independent noise channels
    n_noise_channels = 2 * len(jump_ops)
    
    if n_noise_channels > 0:
        bm = diffrax.VirtualBrownianTree(
            t_eval[0], t_eval[-1], tol=1e-3, 
            shape=(n_noise_channels,), key=key
        )
        term = diffrax.MultiTerm(
            diffrax.ODETerm(drift),
            diffrax.ControlTerm(diffusion, bm)
        )
    else:
        # Closed system fallback
        term = diffrax.ODETerm(drift)
    
    solver = diffrax.Heun()
    
    # 3. Vectorized Solve
    def solve_single(s_init):
        return diffrax.diffeqsolve(
            term, solver, t_eval[0], t_eval[-1], dt0=0.01,
            y0=s_init, args=args, saveat=diffrax.SaveAt(ts=t_eval),
            max_steps=20000
        ).ys
        
    return jax.vmap(solve_single)(s0_batch)


