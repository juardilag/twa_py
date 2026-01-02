import jax.numpy as jnp
import jax

def make_spin_system_functions(hamiltonian_func, jump_ops_funcs):
    """
    Constructs the Drift and Diffusion functions for a specific physical model.
    
    Args:
        hamiltonian_func: f(s, args) -> scalar (Real)
        jump_ops_funcs: List of [ f(s, args) -> scalar (Complex) ]
        
    Returns:
        drift_fn, diffusion_fn (compatible with diffrax)
    """

    def compute_complex_gradient(func, s, args):
        # 1. Define wrappers for Real and Imag parts (both return Reals)
        def real_part_fn(s_in): return jnp.real(func(s_in, args))
        def imag_part_fn(s_in): return jnp.imag(func(s_in, args))
        
        # 2. Differentiate them separately using standard jax.grad
        # grad returns a vector of shape (3,)
        grad_real = jax.grad(real_part_fn)(s)
        grad_imag = jax.grad(imag_part_fn)(s)
        
        # 3. Recombine
        return grad_real + 1j * grad_imag
    
    # --- A. The Drift Function (Deterministic Dynamics) ---
    def drift_fn(t, s, args):
        # 1. Unitary Part: {s, H}
        grad_H = jax.grad(hamiltonian_func, argnums=0)(s, args)
        # EOM: ds/dt = {s, H} = 2 * (grad_H x s)
        unitary_term = 2.0 * jnp.cross(grad_H, s)
        
        # 2. Dissipative Part (Sum over all Jump Ops)
        dissipative_term = jnp.zeros_like(s)
        
        for L_func in jump_ops_funcs:
            # Evaluate L(s) and gradients
            L_val = L_func(s, args)
            L_conj_val = jnp.conjugate(L_val)
            
            grad_L = compute_complex_gradient(L_func, s, args)
            grad_L_conj = jnp.conjugate(grad_L)
            
            # Calculate brackets {s, L*} and {L, s}
            # {s, L*} = 2 * (grad_L* x s)
            bracket_s_Lconj = 2.0 * jnp.cross(grad_L_conj, s)
            
            # {L, s} = - {s, L} = - 2 * (grad_L x s)
            bracket_L_s = -2.0 * jnp.cross(grad_L, s)
            
            # Protocol Drift Term:
            # -(i/2) * [ {s, L*} * L  +  L* * {L, s} ]
            term = (-1j / 2.0) * (bracket_s_Lconj * L_val + L_conj_val * bracket_L_s)
            
            dissipative_term += jnp.real(term)
            
        return unitary_term + dissipative_term

    # --- B. The Diffusion Function (Noise Coupling) ---
    def diffusion_fn(t, s, args):
        # We need to construct a matrix of shape (3, 2 * num_jump_ops)
        # For each jump op, we have Real and Imaginary noise channels
        
        noise_columns = []
        
        for L_func in jump_ops_funcs:
            # Gradients (Same as drift)
            grad_L = compute_complex_gradient(L_func, s, args)
            grad_L_conj = jnp.conjugate(grad_L)
            
            bracket_s_Lconj = 2.0 * jnp.cross(grad_L_conj, s)
            bracket_L_s     = -2.0 * jnp.cross(grad_L, s)
            
            # Noise Terms from Protocol:
            # Term = -(i/2) * [ {s, L*} * xi  +  {L, s} * xi* ]
            # xi = xi_R + i * xi_I
            # Coeff_R = -(i/2) * (Bracket1 + Bracket2)
            # Coeff_I = -(i/2) * (i * Bracket1 - i * Bracket2) = (1/2)*(Bracket1 - Bracket2)
            
            c1 = (-1j / 2.0) * bracket_s_Lconj
            c2 = (-1j / 2.0) * bracket_L_s
            
            col_real = (c1 + c2)
            col_imag = 1j * (c1 - c2)
            
            # We need variance <xi xi*> = 2.
            noise_columns.append(jnp.real(col_real))
            noise_columns.append(jnp.real(col_imag))
            
        if not noise_columns: # Closed System case
            return jnp.zeros((3, 0))
            
        return jnp.stack(noise_columns, axis=-1)

    return drift_fn, diffusion_fn


def make_tavis_cummings_system_functions(omega_c, omega_s, g, kappa, gamma_d, gamma_u, N):
    
    g_eff = g / 2.0 
    g_scaled = g_eff / jnp.sqrt(N)
    
    gamma_diff = 0.5 * (gamma_d - gamma_u)
    sqrt_k = jnp.sqrt(kappa)
    sqrt_gd = jnp.sqrt(gamma_d)
    sqrt_gu = jnp.sqrt(gamma_u)

    def drift_fn(t, y, args):
        a_re, a_im = y[0], y[1]
        s = y[2:].reshape(N, 3)
        sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
        sum_sx, sum_sy = jnp.sum(sx), jnp.sum(sy)
    
        d_a_re = omega_c * a_im - g_scaled * sum_sy - 0.5 * kappa * a_re
        d_a_im = -omega_c * a_re - g_scaled * sum_sx - 0.5 * kappa * a_im
    
        d_sx = -omega_s * sy - 4.0 * g_scaled * sz * a_im + gamma_diff * sx * sz
        d_sy = omega_s * sx - 4.0 * g_scaled * sz * a_re + gamma_diff * sy * sz
        d_sz = 4.0 * g_scaled * (sy * a_re + sx * a_im) - gamma_diff * (sx**2 + sy**2)
        
        return jnp.concatenate([jnp.array([d_a_re, d_a_im]), jnp.stack([d_sx, d_sy, d_sz], axis=1).flatten()])

    def diffusion_fn(t, y, args):
        s = y[2:].reshape(N, 3)
        sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
        n_state, n_noise = 2 + 3 * N, 2 + 4 * N
        B = jnp.zeros((n_state, n_noise))
        
        # Photon Noise
        B = B.at[0, 0].set(-0.5 * sqrt_k); B = B.at[1, 1].set(-0.5 * sqrt_k)
        
        # Spin Noise
        i = jnp.arange(N)
        row_sx, row_sy, row_sz = 2 + 3*i, 2 + 3*i+1, 2 + 3*i+2
        col_dx, col_ux = 2 + 4*i, 2 + 4*i+1
        col_dy, col_uy = 2 + 4*i+2, 2 + 4*i+3
        
        B = B.at[row_sx, col_dx].set(sqrt_gd * sz); B = B.at[row_sx, col_ux].set(-sqrt_gu * sz)
        B = B.at[row_sy, col_dy].set(sqrt_gd * sz); B = B.at[row_sy, col_uy].set(sqrt_gu * sz)
        B = B.at[row_sz, col_ux].set(sqrt_gu * sx); B = B.at[row_sz, col_dx].set(-sqrt_gd * sx)
        B = B.at[row_sz, col_uy].set(-sqrt_gu * sy); B = B.at[row_sz, col_dy].set(-sqrt_gd * sy)
        return B

    return drift_fn, diffusion_fn