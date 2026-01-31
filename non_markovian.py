import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

# Enable 64-bit precision for better numerical stability in physics simulations
jax.config.update("jax_enable_x64", True)

def get_spectral_density(omega, eta, omega_c, s):
    """
    Computes the spectral density J(omega) for the Ohmic family.
    
    Formula: J(omega) = eta * omega * (omega/omega_c)^(s-1) * exp(-omega/omega_c)
    
    Parameters:
    -----------
    omega : jnp.array
        Input frequencies (must be > 0).
    eta : float
        Coupling strength.
    omega_c : float
        Cutoff frequency.
    s : float
        Ohmicity exponent.
        s = 1 : Ohmic
        s < 1 : Sub-Ohmic (e.g., 0.5)
        s > 1 : Super-Ohmic (e.g., 3.0)
        
    Returns:
    --------
    jnp.array
        The spectral density J(omega).
    """
    # We use jnp.where to ensure J(omega) = 0 for omega <= 0 to avoid NaNs with fractional powers
    
    # Calculate the power law part safely
    # (omega / omega_c)^(s-1)
    # Note: If omega is 0 and s < 1, this technically diverges, 
    # but J(omega) ~ omega^s, so as long as s > 0, the total function is 0 at 0.
    
    # safe_omega avoids division by zero or log(0) issues during power calculation
    safe_omega = jnp.where(omega > 0, omega, 1.0) 
    
    prefactor = eta * safe_omega
    power_term = jnp.power(safe_omega / omega_c, s - 1)
    cutoff_term = jnp.exp(-safe_omega / omega_c)
    
    J_val = prefactor * power_term * cutoff_term
    
    # Force 0 where omega <= 0
    return jnp.where(omega > 0, J_val, 0.0)


def compute_correlation_function(times, spectral_density_fn, beta, w_max, n_steps):
    """
    Computes the correlation function C(t) using a Riemann sum integration.
    
    Parameters:
    -----------
    times : jnp.array
        Array of time points t to evaluate C(t).
    spectral_density_fn : function
        A function J(omega) that returns the spectral density array.
    beta : float
        Inverse temperature (1/kT).
    w_max : float
        Maximum frequency for integration limit.
    n_steps : int
        Number of integration steps (Riemann grid size).
        
    Returns:
    --------
    C_t : jnp.array (complex)
        The complex correlation function C(t).
    """
    
    # 1. Discretize Frequency Domain
    # We start slightly above 0 to avoid division by zero in coth(beta*w/2)
    # The contribution at exactly w=0 is usually 0 for Ohmic baths anyway.
    d_omega = w_max / n_steps
    omegas = jnp.linspace(d_omega, w_max, n_steps)
    
    # 2. Precompute Frequency-Dependent Terms (Vectorized)
    # Get J(omega) values
    J_vals = spectral_density_fn(omegas)
    
    # Calculate the Thermal Factor: coth(beta * omega / 2)
    # coth(x) = 1 / tanh(x)
    coth_term = 1.0 / jnp.tanh(beta * omegas / 2.0)
    
    # The integrand prefactor: J(w) * coth(...) for Real part, J(w) for Imag part
    real_prefactor = J_vals * coth_term
    imag_prefactor = J_vals
    
    # 3. Define the Single-Step Integration Function
    # This function calculates C(t) for a *single* time point t
    def integrate_single_time(t):
        # Calculate the oscillating terms for all frequencies at once
        cos_vals = jnp.cos(omegas * t)
        sin_vals = jnp.sin(omegas * t)
        
        # Construct the integrand array
        # Real part: J(w) * coth(bw/2) * cos(wt)
        integrand_real = real_prefactor * cos_vals
        
        # Imaginary part: - J(w) * sin(wt)
        integrand_imag = -imag_prefactor * sin_vals
        
        # Sum over all frequencies (Riemann Sum) and multiply by d_omega
        integral = jnp.sum(integrand_real + 1j * integrand_imag) * d_omega
        return integral

    # 4. Vectorize over Time
    # vmap transforms the function that takes 'float t' into one that takes 'array t'
    C_t = jax.vmap(integrate_single_time)(times)
    
    return C_t


def fit_correlation_function(times, C_target, n_modes=3, learning_rate=0.01, epochs=5000):
    """
    Fits the correlation function C(t) to a sum of exponentials using Optax.
    
    Ansatz: C_model(t) = sum_k c_k * exp(-gamma_k * t)
    
    Parameters:
    -----------
    times : jnp.array
        Time points corresponding to the data.
    C_target : jnp.array (complex)
        The target correlation function values.
    n_modes : int
        Number of exponential modes (auxiliary variables) to use.
    
    Returns:
    --------
    fitted_params : dict
        The optimized parameters (weights c_k and rates gamma_k).
    model_prediction : jnp.array
        The reconstructed C(t) from the fit.
    """
    
    # 1. Initialize Parameters
    # We need complex weights (c) and complex rates (gamma).
    # We store Real/Imag parts separately to enforce constraints easily.
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    params = {
        # Weights c_k = (cr + 1j * ci)
        "c_real": jax.random.normal(k1, (n_modes,)) * 10.0,
        "c_imag": jax.random.normal(k2, (n_modes,)) * 10.0,
        
        # Rates gamma_k = (gr + 1j * gi)
        # We initialize gr positive.
        "gamma_real": jax.random.uniform(k3, (n_modes,), minval=0.1, maxval=2.0),
        "gamma_imag": jax.random.normal(k4, (n_modes,)) * 0.5
    }
    
    # 2. Define the Model (Ansatz)
    def forward(params, t):
        # Enforce Stability: Re[gamma] must be > 0
        # softplus(x) = log(1 + exp(x)) ensures positivity
        gammas = jax.nn.softplus(params["gamma_real"]) + 1j * params["gamma_imag"]
        weights = params["c_real"] + 1j * params["c_imag"]
        
        # Compute sum_k c_k * exp(-gamma_k * t)
        # We use vmap to broadcast over the modes for a single time t
        # But since t is an array, we need to handle shapes carefully.
        
        # Shape: (n_modes, 1) * (1, n_times) -> (n_modes, n_times)
        exponents = jnp.exp(-gammas[:, None] * t[None, :])
        terms = weights[:, None] * exponents
        
        # Sum over modes to get C(t)
        return jnp.sum(terms, axis=0)

    # 3. Define Loss Function (Mean Squared Error on Complex Plane)
    def loss_fn(params, t, target):
        prediction = forward(params, t)
        diff = prediction - target
        # Loss is sum of squared absolute differences
        # |z|^2 = Re(z)^2 + Im(z)^2
        return jnp.mean(jnp.abs(diff)**2)

    # 4. Setup Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # 5. Update Step (JIT Compiled)
    @jax.jit
    def step(params, opt_state, t, target):
        loss, grads = jax.value_and_grad(loss_fn)(params, t, target)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # 6. Training Loop
    print(f"Starting fit with {n_modes} modes...")
    loss_history = []
    
    for i in range(epochs):
        params, opt_state, loss = step(params, opt_state, times, C_target)
        if i % 1000 == 0:
            loss_history.append(loss)
            print(f"Epoch {i}: Loss = {loss:.6e}")
            
    print(f"Final Loss: {loss:.6e}")
    
    # 7. Extract Final Clean Parameters
    final_gammas = jax.nn.softplus(params["gamma_real"]) + 1j * params["gamma_imag"]
    final_weights = params["c_real"] + 1j * params["c_imag"]
    
    clean_params = {
        "gammas": final_gammas,
        "weights": final_weights
    }
    
    final_prediction = forward(params, times)
    
    return clean_params, final_prediction, loss_history