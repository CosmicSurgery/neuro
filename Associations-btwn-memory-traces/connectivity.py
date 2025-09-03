# connectivity.py
"""
Connection masks and delays per Table 1; p(d)=c·exp(-d/λ) for E/I pathways (eq. 7).
E→E and Inp→E are uniform 50%.

Pragmatic note: Table 1 provides scaling constants 'c'. Because PDF parsing of those
'c' values can be ambiguous, we *calibrate* c per pathway so that the *mean* connection
probability matches the ~4–5% averages reported beneath eq. (7), while still yielding
~100% for very small d (local strong inhibition). This preserves the intended geometry-
dependent sparsity without hard-coding potentially misread constants. See paper text
under eq. (7).

Delays are Normal with CV=0.5 and means per Table 1; rounded to integer ms.
"""
import numpy as np
from scipy.sparse import csr_matrix
from config import CFG
from geometry import grid_positions, pairwise_dist

rng = np.random.default_rng(CFG.seed)

def _clip_nonneg_int(x):
    return np.maximum(0, np.rint(x).astype(int))

def _normal_cv(mean, cv, size):
    sigma = cv * mean
    return rng.normal(loc=mean, scale=sigma, size=size)

def _gamma_from_mean_cv(mean, cv, size):
    k = 1.0/(cv**2)
    theta = mean * (cv**2)
    return rng.gamma(shape=k, scale=theta, size=size)

def _calibrate_c(distances, target_mean, lam):
    # Find c so that mean(min(1, c*exp(-d/lam))) ≈ target_mean
    d = distances.ravel()
    def mean_prob(c):
        return np.mean(np.minimum(1.0, c*np.exp(-d/lam)))
    # bisection on log10(c) in [1e-6, 1e+3]
    lo, hi = 1e-6, 1e3
    for _ in range(40):
        mid = (lo+hi)/2
        if mean_prob(mid) > target_mean:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2

def build_masks_and_delays():
    N_inp, N_E, N_I = CFG.N_inp, CFG.N_E, CFG.N_I
    N = N_E + N_I
    coords = grid_positions()
    D = pairwise_dist(coords)

    # Index partitions: first N_E are excitatory, then N_I inhibitory
    E_idx = np.arange(N_E)
    I_idx = np.arange(N_E, N_E+N_I)

    # Inp→E (uniform 50%)
    mask_inpE = rng.random((N_E, N_inp)) < CFG.p_inpE  # postsyn rows × presyn cols

    # E→E (uniform 50%), no self
    mask_EE = rng.random((N_E, N_E)) < CFG.p_EE
    np.fill_diagonal(mask_EE, False)

    # Distance-dependent p(d)=c*exp(-d/lambda) for E→I, I→E, I→I
    lam = CFG.lambda_dist
    # Extract sub-distance matrices
    D_EI = D[np.ix_(I_idx, E_idx)]  # postsyn I × presyn E
    D_IE = D[np.ix_(E_idx, I_idx)]  # postsyn E × presyn I
    D_II = D[np.ix_(I_idx, I_idx)]
    np.fill_diagonal(D_II, np.inf)  # prevent self

    c_EI = _calibrate_c(D_EI, CFG.target_mean_p_EI, lam)
    c_IE = _calibrate_c(D_IE, CFG.target_mean_p_IE, lam)
    c_II = _calibrate_c(D_II, CFG.target_mean_p_II, lam)

    P_EI = np.minimum(1.0, c_EI*np.exp(-D_EI/lam))
    P_IE = np.minimum(1.0, c_IE*np.exp(-D_IE/lam))
    P_II = np.minimum(1.0, c_II*np.exp(-D_II/lam))

    mask_EI = rng.random(P_EI.shape) < P_EI
    mask_IE = rng.random(P_IE.shape) < P_IE
    mask_II = rng.random(P_II.shape) < P_II
    np.fill_diagonal(mask_II, False)

    # Delays per Table 1
    def delays_for_mask(mask, mean_ms):
        d = _normal_cv(mean_ms, CFG.d_cv, mask.shape).astype(float)
        d = _clip_nonneg_int(d)
        d[~mask] = 0
        return d

    d_inpE = delays_for_mask(mask_inpE, CFG.dmean_inpE_ms)
    d_EE   = delays_for_mask(mask_EE,   CFG.dmean_EE_ms)
    d_EI   = delays_for_mask(mask_EI,   CFG.dmean_EI_ms)
    d_IE   = delays_for_mask(mask_IE,   CFG.dmean_IE_ms)
    d_II   = delays_for_mask(mask_II,   CFG.dmean_II_ms)

    return dict(
        mask_inpE=mask_inpE, d_inpE=d_inpE,
        mask_EE=mask_EE,     d_EE=d_EE,
        mask_EI=mask_EI,     d_EI=d_EI,
        mask_IE=mask_IE,     d_IE=d_IE,
        mask_II=mask_II,     d_II=d_II,
        c_constants=dict(EI=c_EI, IE=c_IE, II=c_II)
    )
