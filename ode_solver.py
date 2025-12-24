import jax
import jax.numpy as jnp
import diffrax

def solve_twa_dynamics(
    state_initial : jnp.ndarray, 
    t_array : jnp.ndarray, 
    eom_func,  
    system_args : list, 
    solver=diffrax.Dopri5()):
    """
    A General TWA Solver for any Hamiltonian/Equation of Motion.
    
    Args:
        state_initial (jax.Array): Initial batch of states (N_traj, State_Dim).
        t_array (jax.Array): Array of time points to save.
        eom_func (callable): Function f(state, t, args) -> dstate/dt.
                             Must be written for a SINGLE trajectory.
        system_args (PyTree): Parameters to pass to eom_func (e.g., Omega, couplings).
        solver (diffrax.AbstractSolver): The ODE solver (default: Dopri5).
        
    Returns:
        jax.Array: Trajectories of shape (n_times, n_trajectories, state_dim).
    """
    
    # 1. Define the Vector Field (The "Driver")
    # This wrapper handles the parallelization (vmap) automatically.
    # It assumes eom_func signature is: eom_func(state, t, args)
    def vector_field(t, state_batch, args):
        # jax.vmap applies eom_func to every trajectory in the batch at once.
        return jax.vmap(eom_func, in_axes=(0, None, None))(state_batch, t, args)

    # 2. Setup Solver Options
    state_initial = state_initial.astype(jnp.float32)
    t_array = t_array.astype(jnp.float32)
    
    term = diffrax.ODETerm(vector_field)
    t0 = t_array[0]
    t1 = t_array[-1]
    saveat = diffrax.SaveAt(ts=t_array)
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    
    # 3. Solve the Differential Equation
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0=None,              # Let solver pick initial step size
        y0=state_initial,      # The swarm of initial conditions
        args=system_args,      # Pass the general arguments here
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=16**4
    )
    
    return sol.ys