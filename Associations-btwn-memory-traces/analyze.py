# analyze.py
"""
Assembly (PRU) detection and PCUs (Sections 2.2.4 & 2.3.2).
- PRUs: Wilcoxon rank-sum (Mann–Whitney U) test between baseline [-100,0] ms and response [10,110] ms;
        require median firing ≥ 2 Hz in response window.
- PCUs: Among PRUs with a single preferred (P) component (blue or green), units that gain significance
        to their nonpreferred (NP) after association, with single-trial increases NP > NA (Wilcoxon).

Implementation: lightweight Mann–Whitney U with normal approx (no SciPy dependency).
"""
import numpy as np
from math import sqrt

def mannwhitney_u(x, y):
    # Two-sided p-value via normal approximation with continuity correction.
    n1, n2 = len(x), len(y)
    if n1==0 or n2==0: return 1.0
    allv = np.concatenate([x, y])
    ranks = allv.argsort().argsort().astype(float)+1
    R1 = ranks[:n1].sum()
    U1 = R1 - n1*(n1+1)/2.0
    mu = n1*n2/2.0
    sigma = sqrt(n1*n2*(n1+n2+1)/12.0)
    z = (U1 - mu - 0.5*np.sign(U1-mu))/sigma if sigma>0 else 0.0
    # 2-sided
    from math import erf
    p = 2.0*(1.0 - 0.5*(1.0+erf(abs(z)/sqrt(2))))
    return p

def window_rates(spike_times_ms, trial_onsets_ms, w0, w1, T_ms):
    """Return array [n_trials] of spike counts in [w0,w1] per trial, converted to Hz."""
    counts = []
    for t0 in trial_onsets_ms:
        a, b = t0+w0, t0+w1
        cnt = np.sum((spike_times_ms >= a) & (spike_times_ms < b))
        counts.append(cnt * (1000.0 / (w1 - w0)))
    return np.array(counts, dtype=float)

def detect_PRUs(E_spike_trains, trial_onsets, labels):
    """
    E_spike_trains: list length N_E, each is sorted np.array of ms spike times.
    trial_onsets: dict label -> list of onset ms for that pattern.
    Returns: dict with fields 'assemblies' (label->list of neuron idx), 'PRUs' (set).
    """
    N_E = len(E_spike_trains)
    assemblies = {lab: [] for lab in labels}
    PRUs = set()
    for i in range(N_E):
        spike_t = E_spike_trains[i]
        for lab in labels:
            base = window_rates(spike_t, trial_onsets[lab], -100, 0, 110)
            resp = window_rates(spike_t, trial_onsets[lab], 10, 110, 110)
            p = mannwhitney_u(base, resp)
            med_hz = np.median(resp)
            if p < 0.05 and med_hz >= 2.0:
                assemblies[lab].append(i); PRUs.add(i)
                break  # first preferred wins
    return dict(assemblies=assemblies, PRUs=PRUs)

def detect_PCUs(E_spike_trains_before, E_spike_trains_after, trial_onsets_before, trial_onsets_after, assemblies):
    """
    Return: number of PCUs and list of indices.
    """
    blue, green, red = "blue", "green", "red"
    PRUs_single = []
    # choose PRUs that belong to exactly one of blue/green
    for i in set(assemblies[blue]+assemblies[green]+assemblies[red]):
        flags = [i in assemblies[blue], i in assemblies[green]]
        if sum(flags)==1:
            PRUs_single.append(i)
    PCUs = []
    for i in PRUs_single:
        P = blue if (i in assemblies[blue]) else green
        NP = green if P==blue else blue
        NA = red
        # before
        baseB = window_rates(E_spike_trains_before[i], trial_onsets_before[NP], -100, 0, 110)
        respB = window_rates(E_spike_trains_before[i], trial_onsets_before[NP], 10, 110, 110)
        p_before = mannwhitney_u(baseB, respB)
        # after
        baseA = window_rates(E_spike_trains_after[i], trial_onsets_after[NP], -100, 0, 110)
        respA = window_rates(E_spike_trains_after[i], trial_onsets_after[NP], 10, 110, 110)
        p_after = mannwhitney_u(baseA, respA)
        if not (p_before >= 0.05 and p_after < 0.05):
            continue
        # single-trial increase NP vs NA (after > before; NP increases more than NA)
        inc_NP = respA - window_rates(E_spike_trains_before[i], trial_onsets_before[NP], 10, 110, 110).mean()
        inc_NA = window_rates(E_spike_trains_after[i], trial_onsets_after[NA], 10, 110, 110) - \
                 window_rates(E_spike_trains_before[i], trial_onsets_before[NA], 10, 110, 110).mean()
        p_inc = mannwhitney_u(inc_NP, inc_NA)
        if p_inc < 0.05:
            PCUs.append(i)
    return len(PCUs), PCUs
