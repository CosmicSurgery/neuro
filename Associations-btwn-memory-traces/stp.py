# stp.py
"""
Short-term plasticity (STP) state updates per presynaptic neuron & pathway (eqs. 10–14).
Per presyn j, maintain u_j, R_j; on each spike:
  - Use eff_j = R_j * u_j  as the multiplicative dynamic factor for all outgoing synapses of that pathway.
  - Then update (u,R) with Δt = time since last spike of that *presyn* (per pathway), as in eqs. (12)–(14).

Bounds: U ∈ [0.001, 0.999], D,F ∈ [0.1, 5000] ms (text around Table 2).
"""
import numpy as np

class STPState:
    def __init__(self, U, D, F):
        self.U = U.astype(np.float32)
        self.D = D.astype(np.float32)
        self.F = F.astype(np.float32)
        n = U.shape[0]
        self.u = self.U.copy()
        self.R = 1.0 - self.U  # eq. (11)
        self.last_t = np.zeros(n, dtype=np.int64)

    def on_spikes(self, t_ms, spikes_idx):
        """Return eff factors R*u for presyn indices, then update their states."""
        if len(spikes_idx) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=int)
        dt = np.maximum(1.0, (t_ms - self.last_t[spikes_idx]).astype(np.float32))
        # Update u (eq. 14) and R (eq. 12) using dt in ms
        u_prev = self.u[spikes_idx]
        R_prev = self.R[spikes_idx]
        u_new = u_prev * np.exp(-dt/self.F[spikes_idx]) + self.U[spikes_idx]*(1.0 - u_prev*np.exp(-dt/self.F[spikes_idx]))
        R_new = R_prev*(1.0 - u_new)*np.exp(-dt/self.D[spikes_idx]) + 1.0 - np.exp(-dt/self.D[spikes_idx])

        eff = (R_new * u_new).astype(np.float32)  # to be applied to outgoing
        # Commit updates; last_t to current time
        self.u[spikes_idx] = u_new
        self.R[spikes_idx] = R_new
        self.last_t[spikes_idx] = t_ms
        return eff, spikes_idx
