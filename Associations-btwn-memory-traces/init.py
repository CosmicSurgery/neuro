# init.py
"""
Initial weights (Gamma, CV=0.7) and STP parameters with steady-state correction (eqs. 10–18).

- Weights μ: Inp→E=γ_w*15, E→E=γ_w*2.5, E→I=1000, I→E=1375, I→I=6000 (Table 1).
- STP UDF: bounded Gamma around Table 2 means±SD. For E→E: F≈0, clamped to bound [0.1, 5000] ms.
- Steady-state correction: adjust absolute efficacies A so that dynamic steady-state at f0=5 Hz equals sampled w_init (eqs. 15–18).

Pragmatic note: STP states are tracked per *presynaptic neuron & pathway type* (not per individual synapse)
to keep the reference code compact; the data constraints (Table 2) remain respected.
"""
import numpy as np
from config import CFG
from connectivity import _gamma_from_mean_cv, rng

def _bounded_gamma(mean, sd, size, lo, hi):
    if sd == 0.0:
        out = np.full(size, mean, dtype=np.float32)
    else:
        k = (mean/sd)**2
        theta = (sd**2)/mean
        out = rng.gamma(k, theta, size=size).astype(np.float32)
    return np.clip(out, lo, hi)

def sample_weights_and_stp(masks):
    N_inp, N_E, N_I = CFG.N_inp, CFG.N_E, CFG.N_I

    # Weights per connection matrix (post×pre)
    def W_from_mask(mask, mu):
        w = _gamma_from_mean_cv(mu, CFG.w_cv, mask.shape).astype(np.float32)
        w *= mask
        return w

    W_inpE = W_from_mask(masks["mask_inpE"], CFG.gamma_w*CFG.mu_inpE)
    W_EE   = W_from_mask(masks["mask_EE"],   CFG.gamma_w*CFG.mu_EE)
    W_EI   = W_from_mask(masks["mask_EI"],   CFG.mu_EI)
    W_IE   = W_from_mask(masks["mask_IE"],   CFG.mu_IE)
    W_II   = W_from_mask(masks["mask_II"],   CFG.mu_II)

    # Per-presyn UDF params for each pathway type
    # Sizes are number of presyn neurons for that type
    UDF = {}
    for typ, n_pre in [("EE", CFG.N_E), ("EI", CFG.N_E), ("IE", CFG.N_I), ("II", CFG.N_I)]:
        means, sds = CFG.stp_means[typ], CFG.stp_sds[typ]
        U = _bounded_gamma(means["U"], sds["U"], n_pre, *CFG.stp_bounds_U)
        D = _bounded_gamma(means["D"], sds["D"], n_pre, *CFG.stp_bounds_D)
        F = _bounded_gamma(max(means["F"], CFG.stp_bounds_F[0]), sds["F"], n_pre, *CFG.stp_bounds_F)
        UDF[typ] = dict(U=U.astype(np.float32), D=D.astype(np.float32), F=F.astype(np.float32))

    # Steady-state correction A' = w_init / (R*(f0) * u*(f0)) for dynamic synapses (eqs. 15–18)
    def ss_correction(U, D, F):
        r = CFG.f0_hz_for_ss
        u_star = U / (1.0 - (1.0 - U)*np.exp(-1.0/(r*F)))
        R_star = (1.0 - np.exp(-1.0/(r*D))) / (1.0 - (1.0 - u_star)*np.exp(-1.0/(r*D)))
        return (R_star * u_star).astype(np.float32)

    # Apply per pathway to absolute efficacies (weights act as A)
    # Inp→E has no STP params in Table 2 => no correction there.
    for (W, typ, n_pre) in [(W_EE, "EE", CFG.N_E), (W_EI, "EI", CFG.N_E),
                            (W_IE, "IE", CFG.N_I), (W_II, "II", CFG.N_I)]:
        U, D, F = UDF[typ]["U"], UDF[typ]["D"], UDF[typ]["F"]
        corr = ss_correction(U, D, F)  # shape (n_pre,)
        # Scale columns (per presyn) so that dynamic steady-state equals sampled W
        # A'_{:,j} = W_{:,j} / corr_j
        scale = np.ones_like(U, dtype=np.float32)
        nz = corr > 1e-12
        scale[nz] = 1.0 / corr[nz]
        W *= scale[None, :]

    return dict(W_inpE=W_inpE, W_EE=W_EE, W_EI=W_EI, W_IE=W_IE, W_II=W_II, UDF=UDF)
