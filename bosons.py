import jax.numpy as jnp
import jax

def make_boson_system_functions(hamiltonian_func, jump_ops_funcs):
    """
    Constructs Drift/Diffusion for Bosonic Modes (Complex Scalar Fields).
    State 'psi' is a complex array of shape (N_modes,).
    
    Equations:
      d(psi)/dt = {psi, H} 
                  - i/2 * {psi, L*} * (L + xi) 
                  - i/2 * (L* + xi*) * {L, psi}
    
    Boson Poisson Bracket:
      {psi_i, psi_j*} = -i * delta_ij
      {f, g} = -i * (df/dpsi * dg/dpsi* - df/dpsi* * dg/dpsi)
      
    Key simplification:
      {psi, G} = -i * dG/dpsi*
    """

    # Helper: JAX grad(f)(z) returns (df/dx + i df/dy) = 2 * df/dz*
    # So: df/dz* = 0.5 * grad(f)
    def get_df_dpsi_star(func, psi, args):
        def real_fn(p): return jnp.real(func(p, args))
        def imag_fn(p): return jnp.imag(func(p, args))
        # Linearity: d/dz* (u + iv) = du/dz* + i dv/dz*
        grad_u = jax.grad(real_fn)(psi)
        grad_v = jax.grad(imag_fn)(psi)
        return 0.5 * (grad_u + 1j * grad_v)

    def drift_fn(t, psi, args):
        # 1. Unitary Part: {psi, H} = -i * dH/dpsi*
        # H is real, so jax.grad(H) is exactly 2*dH/dpsi*
        grad_H = jax.grad(hamiltonian_func, argnums=0)(psi, args)
        dH_dpsi_star = 0.5 * grad_H
        unitary_term = -1j * dH_dpsi_star
        
        # 2. Dissipative Part
        dissipative_term = jnp.zeros_like(psi)
        
        for L_func in jump_ops_funcs:
            L_val = L_func(psi, args)
            L_conj = jnp.conjugate(L_val)
            
            # Calculate Derivatives
            # dL*/dpsi*
            dL_star_dpsi_star = get_df_dpsi_star(lambda p, a: jnp.conjugate(L_func(p, a)), psi, args)
            # dL/dpsi* (Usually 0 for standard L=psi, but kept for generality)
            dL_dpsi_star = get_df_dpsi_star(L_func, psi, args)
            
            # Calculate Poisson Brackets
            # {psi, L*} = -i * dL*/dpsi*
            bracket_psi_Lstar = -1j * dL_star_dpsi_star
            
            # {L, psi} = -{psi, L} = -(-i * dL/dpsi*) = i * dL/dpsi*
            # Note: TWA usually involves {L, psi}, but formula has {L, psi} * L*
            # Let's use the identity: {L, psi} = - {psi, L}
            # {psi, L} = -i * dL/dpsi*
            bracket_L_psi = 1j * dL_dpsi_star
            
            # Drift Formula: -i/2 * {psi, L*} * L  - i/2 * L* * {L, psi}
            term = (-1j / 2.0) * (bracket_psi_Lstar * L_val + L_conj * bracket_L_psi)
            dissipative_term += term
            
        return unitary_term + dissipative_term

    def diffusion_fn(t, psi, args):
        noise_columns = []
        
        for L_func in jump_ops_funcs:
            # Re-calculate derivatives
            dL_star_dpsi_star = get_df_dpsi_star(lambda p, a: jnp.conjugate(L_func(p, a)), psi, args)
            dL_dpsi_star = get_df_dpsi_star(L_func, psi, args)
            
            bracket_psi_Lstar = -1j * dL_star_dpsi_star
            bracket_L_psi     = 1j * dL_dpsi_star
            
            # Noise Coefficients (Same -i/2 factor)
            # Term: -i/2 * {psi, L*} * xi  - i/2 * {L, psi} * xi*
            c1 = (-1j / 2.0) * bracket_psi_Lstar
            c2 = (-1j / 2.0) * bracket_L_psi
            
            # Split Complex Noise xi = xi_R + i xi_I
            # Term: c1(R + iI) + c2(R - iI) = (c1+c2)R + i(c1-c2)I
            col_real = (c1 + c2)
            col_imag = 1j * (c1 - c2)
            
            # Append columns (No extra scaling, assumes L includes sqrt(gamma))
            noise_columns.append(col_real)
            noise_columns.append(col_imag)
            
        if not noise_columns:
            return jnp.zeros((psi.shape[0], 0), dtype=psi.dtype)
            
        return jnp.stack(noise_columns, axis=-1)

    return drift_fn, diffusion_fn