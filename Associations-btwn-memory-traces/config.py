# config.py
"""
Config and global defaults for the Pokorny et al. (2020) model.

Cross-checks:
- Kernel eq. (1), rate eq. (2), membrane eq. (3).
- Geometry, probabilities, weights, delays per Table 1.
- STP eqs. (10)–(14) + steady-state correction eqs. (15)–(18).
- Triplet STDP rule eq. (19) + clipping caps.
- Input protocol and 5-phase schedule (Sections 2.2–2.3; Fig. 2).

Any pragmatic deviations are called out in module docstrings.
"""
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 7

    # Network sizes (Section 2.1; Fig. 1A)
    N_inp: int = 200
    N_E: int = 432
    N_I: int = 108

    # Geometry (Section 2.1; 3D grid 6×6×15)
    grid_shape = (6, 6, 15)  # 6*6*15 = 540

    # PSP kernel (eq. 1)
    tau_r_ms: float = 2.0
    tau_f_ms: float = 20.0
    T_eps_ms: int = 100

    # Transfer function (eq. 2)
    r0_hz: float = 1.238
    rate_exp_coeff: float = 0.25
    max_u_for_exp: float = 400.0  # prevent overflow

    # Membrane offsets (eq. 3 & surrounding text)
    exc_log_mu: float = 2.64  # underlying normal mu
    exc_log_sigma: float = 0.23
    exc_log_shift: float = -600.0
    gamma_exc: float = 0.8
    E_exc_generic: float = 300.0  # E_exc,generic before scaling (eq. 5)
    I_exc_generic: float = 450.0  # inhibitory generic

    # Refractory (Gamma k=2) means (E/I) in ms
    refrac_mean_E_ms: float = 10.0
    refrac_mean_I_ms: float = 3.0

    # Connectivity (Table 1)
    p_EE: float = 0.5  # uniform E→E
    p_inpE: float = 0.5  # uniform Inp→E
    lambda_dist: float = 0.25  # for distance-dependent p(d)=c*exp(-d/lambda)
    # For E↔I we target mean probs ≈4–5% as reported under eq. (7) paragraph.
    target_mean_p_EI: float = 0.04
    target_mean_p_IE: float = 0.05
    target_mean_p_II: float = 0.04

    # Initial weights (Gamma CV=0.7) (Table 1; eqs. 8–9)
    mu_inpE: float = 15.0  # scaled by gamma_w
    mu_EE: float = 2.5     # scaled by gamma_w
    mu_EI: float = 1000.0
    mu_IE: float = 1375.0
    mu_II: float = 6000.0
    w_cv: float = 0.7

    # Delays (Normal CV=0.5) means from Table 1 (rounded to ms)
    dmean_inpE_ms: float = 5.0
    dmean_EE_ms: float = 5.0
    dmean_EI_ms: float = 2.0
    dmean_IE_ms: float = 2.0
    dmean_II_ms: float = 2.0
    d_cv: float = 0.5

    # STP UDF (Table 2) — bounds per text; E→E depressing; others as given
    stp_bounds_U = (0.001, 0.999)
    stp_bounds_D = (0.1, 5000.0)   # ms
    stp_bounds_F = (0.1, 5000.0)   # ms
    stp_means = {
        "EE": dict(U=0.45, D=144.0, F=0.0),     # depressing (Testa-Silva 2014)
        "EI": dict(U=0.09, D=138.0, F=670.0),   # facilitating
        "IE": dict(U=0.16, D=45.0,  F=376.0),   # facilitating
        "II": dict(U=0.25, D=706.0, F=21.0),    # depressing
    }
    stp_sds = {
        "EE": dict(U=0.17, D=67.0,  F=0.0),
        "EI": dict(U=0.12, D=211.0, F=830.0),
        "IE": dict(U=0.10, D=21.0,  F=253.0),
        "II": dict(U=0.13, D=405.0, F=9.0),
    }
    f0_hz_for_ss: float = 5.0  # steady-state correction rate (eqs. 15–18)

    # STDP (eq. 19) — triplet rule
    tau_r1_ms: float = 25.0
    tau_r2_ms: float = 25.0
    tau_o1_ms: float = 1000.0
    tau_o2_ms: float = 25.0
    A2_plus: float = 10.0
    A2_minus: float = 0.5
    A3_plus: float = 10.0
    A3_minus: float = 0.5
    # Relative caps (text below eq. 19)
    cap_inpE: float = 2.0
    cap_EE: float = 10.0

    # Inputs & schedule (Sections 2.2–2.3; Fig. 2)
    pattern_ms: int = 100
    on_channels: int = 20
    ron_hz: float = 40.0
    roff_hz: float = 0.0
    bg_in_pattern_hz: float = 3.0
    bg_noise_hz: float = 5.0
    max_overlap: int = 2

    # 5 phases durations (s). FAST_DEMO overrides these.
    T_init_s: float = 250.0
    T_encode_s: float = 250.0
    T_rest1_s: float = 250.0
    T_assoc_s: float = 36.97  # average ~20 combined patterns
    T_rest2_s: float = 250.0
    FAST_DEMO: bool = True
    # Demo shrink factors (keep ratios; still exercises full pipeline)
    demo_scale: float = 0.005  # ~2%

    gamma_w: float = 0.36

    scale_IE = 1.05   # scales I→E weights only; 1.0 = paper default
    scale_II = 0   # (leave at 1.0 unless I still saturates)


CFG = Config()
