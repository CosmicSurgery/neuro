# simulate.py
"""
Simulation orchestrator for the 5 phases (Fig. 2), with seeding and light checkpointing.

Core loop (1 ms dt), cross-checked to paper items:
- PSP kernel: double-exponential with τ_r=2 ms, τ_f=20 ms, cutoff 100 ms, peak=1 (eq. 1).
- Membrane u and stochastic spiking with r(t)=r0·exp(0.25·u) (eqs. 2–3).
- Geometry/connectivity per Table 1; delays are applied to synaptic transmission.
- STP per presyn pathway (eqs. 10–14); steady-state weight correction already applied in init.py (eqs. 15–18).
- Triplet STDP on Inp→E and E→E (eq. 19): updates at emission-time; updates computed before r2/o2 increments; caps enforced.

Implementation notes:
- We use per-pathway ring buffers of presynaptic *effective amplitudes* (after STP) indexed by integer delays.
- Arriving current at t is Σ_d (W∘mask_d) @ v_presyn(t−d). We precompute delay bucket (rows, cols).
- STDP traces are maintained per pre/post unit (simplifies arrival-time bookkeeping; qualitatively faithful).
- STDP is applied to the same dense weight arrays that drive currents, so plastic changes take effect immediately.

Outputs:
- E/I spike trains (ms), final & initial weights, snapshots (optional), schedule, onsets for patterns, and channel sets.
"""
import numpy as np
from collections import defaultdict
from config import CFG
from connectivity import build_masks_and_delays
from init import sample_weights_and_stp
from psp import PSPState
from neurons import NeuronPop
from stp import STPState
from stdp_triplet import TripletSTDP
from inputs import make_three_patterns, build_schedule, generate_input_spikes

# progress bar (optional)
try:
    from tqdm.auto import tqdm  # great in notebooks/terminals
except Exception:
    class tqdm:  # no-op fallback
        def __init__(self, total=None, desc=None, unit=None, leave=False): self.total=total; self.n=0
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): self.n += n
        def close(self): pass


# ---------------------------
# Helpers (no SciPy required)
# ---------------------------

def build_delay_buckets(mask: np.ndarray, delays: np.ndarray):
    """
    Return dict delay(int)->(rows, cols) for all synapses of that delay.
    mask: (post×pre) boolean
    delays: (post×pre) int (ms), valid only where mask==True
    """
    assert mask.shape == delays.shape
    out = {}
    if not np.any(mask):
        return out
    valid_delays = np.unique(delays[mask])
    for d in valid_delays:
        sel = (delays == d) & mask
        rows, cols = np.where(sel)
        if rows.size:
            out[int(d)] = (rows.astype(np.int32), cols.astype(np.int32))
    return out

def weighted_arrivals(W: np.ndarray, delay_buckets: dict, ring: list, t_abs: int, max_delay: int):
    """
    Current arriving to postsyn at time t_abs:
      cur_i = Σ_d Σ_k∈bucket_d,rows==i  W[i,k] * v_presyn_k(t_abs - d)
    W: (post×pre) dense float32 array
    delay_buckets: dict d -> (rows, cols) index arrays into W
    ring: list length (max_delay+1) of presyn vectors (amplitudes after STP), dtype float32
    """
    if not delay_buckets:
        return np.zeros(W.shape[0], dtype=np.float32)
    cur = np.zeros(W.shape[0], dtype=np.float32)
    for d, (rows, cols) in delay_buckets.items():
        v = ring[(t_abs - d) % (max_delay + 1)]
        if v.size == 0 or not np.any(v):
            continue
        # contrib per-synapse: W[rows, cols] * v[cols]
        np.add.at(cur, rows, (W[rows, cols] * v[cols]).astype(np.float32))
    return cur

# ---------------------------
# Main entry
# ---------------------------

def run(progress: bool = False):
    # --- Build network structure & parameters
    masks = build_masks_and_delays()
    weights_stp = sample_weights_and_stp(masks)  # returns dense weight matrices + UDF params
    weights_stp["W_IE"] *= getattr(CFG, "scale_IE", 1.0)
    weights_stp["W_II"] *= getattr(CFG, "scale_II", 1.0)


    # Delay buckets for each pathway
    B_inpE = build_delay_buckets(masks["mask_inpE"], masks["d_inpE"])
    B_EE   = build_delay_buckets(masks["mask_EE"],   masks["d_EE"])
    B_EI   = build_delay_buckets(masks["mask_EI"],   masks["d_EI"])
    B_IE   = build_delay_buckets(masks["mask_IE"],   masks["d_IE"])
    B_II   = build_delay_buckets(masks["mask_II"],   masks["d_II"])

    all_delays = []
    for B in (B_inpE, B_EE, B_EI, B_IE, B_II):
        if B: all_delays.extend(list(B.keys()))
    max_delay = max(all_delays) if all_delays else 0

    # STDP handler (Inp→E and E→E) references the same weight arrays used below
    stdp = TripletSTDP(weights_stp["W_inpE"], weights_stp["W_EE"])

    # STP states per pathway (per presynaptic neuron)
    stp_EE = STPState(**weights_stp["UDF"]["EE"])
    stp_EI = STPState(**weights_stp["UDF"]["EI"])
    stp_IE = STPState(**weights_stp["UDF"]["IE"])
    stp_II = STPState(**weights_stp["UDF"]["II"])

    # Neuron populations & PSP accumulators
    pops = NeuronPop(CFG.N_E, CFG.N_I)
    psp_inp_toE = PSPState(CFG.N_E)
    psp_E_toE   = PSPState(CFG.N_E)
    psp_I_toE   = PSPState(CFG.N_E)
    psp_E_toI   = PSPState(CFG.N_I)
    psp_I_toI   = PSPState(CFG.N_I)

    # Inputs and schedule
    patterns = make_three_patterns()
    schedule = build_schedule(patterns)
    inp_spikes = generate_input_spikes(schedule)

    # STDP adjacency lists
    post_targets_for_inp = {j: np.where(masks["mask_inpE"][:, j])[0].astype(np.int32) for j in range(CFG.N_inp)}
    post_targets_for_E   = {j: np.where(masks["mask_EE"][:, j])[0].astype(np.int32)   for j in range(CFG.N_E)}
    pre_incoming_inp     = {i: np.where(masks["mask_inpE"][i, :])[0].astype(np.int32) for i in range(CFG.N_E)}
    pre_incoming_E       = {i: np.where(masks["mask_EE"][i, :])[0].astype(np.int32)   for i in range(CFG.N_E)}

    # Spike recordings
    E_spike_times = [[] for _ in range(CFG.N_E)]
    I_spike_times = [[] for _ in range(CFG.N_I)]

    # Trial onsets for analyses
    onsets = defaultdict(list)  # label -> list of onset ms (absolute)

    # Per-pathway ring buffers of presyn amplitudes (already STP-modulated; inputs are binary 0/1)
    ring_inp  = [np.zeros(CFG.N_inp, dtype=np.float32) for _ in range(max_delay+1)]
    ring_E_EE = [np.zeros(CFG.N_E,   dtype=np.float32) for _ in range(max_delay+1)]  # E→E
    ring_E_EI = [np.zeros(CFG.N_E,   dtype=np.float32) for _ in range(max_delay+1)]  # E→I
    ring_I_IE = [np.zeros(CFG.N_I,   dtype=np.float32) for _ in range(max_delay+1)]  # I→E
    ring_I_II = [np.zeros(CFG.N_I,   dtype=np.float32) for _ in range(max_delay+1)]  # I→I

    # Minimal snapshots: weights at segment boundaries (keyed by absolute time in ms)
    snapshots = {}

    # --- Time loop over schedule
    t0 = 0
    T_total = sum(seg["T_ms"] for seg in schedule)
    
    # progress bar setup (update ~1000 times to keep overhead low)
    pbar = None
    if progress:
        tick = max(1, T_total // 1000)  # ~1000 refreshes
        pbar = tqdm(total=T_total, desc="Simulating", unit="ms", leave=False)
        done = 0
        
    for seg in schedule:
        if seg["kind"] in ("pattern", "combined") and seg["label"] is not None:
            onsets[seg["label"]].append(t0)

        for t in range(seg["T_ms"]):
            t_abs = t0 + t

            # Decay STDP traces per ms
            stdp.decay()

            # --- Input spikes at this ms (emission-time)
            ring_inp[t_abs % (max_delay+1)].fill(0.0)
            inp_now = inp_spikes.get(t_abs, np.empty(0, dtype=int))
            if inp_now.size > 0:
                ring_inp[t_abs % (max_delay+1)][inp_now] = 1.0
                # STDP pre for these input channels
                stdp.pre_update(pre_inp_idx=inp_now, pre_E_idx=None,
                                post_targets_for_inp=post_targets_for_inp, post_targets_for_E=None)

            # --- Arrivals at t_abs: currents per pathway (weight × delayed presyn amplitude)
            inp_cur = weighted_arrivals(weights_stp["W_inpE"], B_inpE, ring_inp,  t_abs, max_delay)  # onto E
            EE_cur  = weighted_arrivals(weights_stp["W_EE"],   B_EE,   ring_E_EE, t_abs, max_delay)  # E→E
            EI_cur  = weighted_arrivals(weights_stp["W_EI"],   B_EI,   ring_E_EI, t_abs, max_delay)  # E→I
            IE_cur  = weighted_arrivals(weights_stp["W_IE"],   B_IE,   ring_I_IE, t_abs, max_delay)  # I→E
            II_cur  = weighted_arrivals(weights_stp["W_II"],   B_II,   ring_I_II, t_abs, max_delay)  # I→I

            # --- PSP updates (double-exponential; eq. 1)
            y_inp_toE = psp_inp_toE.step(inp_cur)
            zE_toE    = psp_E_toE.step(EE_cur)
            hI_toE    = psp_I_toE.step(IE_cur)
            zE_toI    = psp_E_toI.step(EI_cur)
            hI_toI    = psp_I_toI.step(II_cur)

            # --- Membrane + stochastic spikes (eqs. 2–3)
            spikesE, spikesI = pops.step(y_inp_toE, zE_toE, hI_toE, zE_toI, hI_toI)

            # Record spikes
            if spikesE.any():
                idx = np.where(spikesE)[0]
                for i in idx: E_spike_times[i].append(t_abs)
            if spikesI.any():
                idx = np.where(spikesI)[0]
                for i in idx: I_spike_times[i].append(t_abs)

            # --- STDP post at E spikes (emission-time)
            E_idx = np.where(spikesE)[0]
            if E_idx.size > 0:
                stdp.post_update(post_E_idx=E_idx,
                                 pre_incoming_inp=pre_incoming_inp,
                                 pre_incoming_E=pre_incoming_E)
            # ALSO do the pre-spike update for E→E on E emitters (triplet “pre” term)
            if E_idx.size > 0:
                stdp.pre_update(pre_inp_idx=None, pre_E_idx=E_idx,
                                post_targets_for_inp=None, post_targets_for_E=post_targets_for_E)

            # --- STP emission: schedule pathway-specific presyn amplitudes into rings
            slot = t_abs % (max_delay+1)

            # E presyn rings (E→E, E→I)
            ring_E_EE[slot].fill(0.0)
            ring_E_EI[slot].fill(0.0)
            if E_idx.size > 0:
                eff_EE, idxE = stp_EE.on_spikes(t_abs, E_idx)
                eff_EI, _    = stp_EI.on_spikes(t_abs, E_idx)
                vEE = np.zeros(CFG.N_E, dtype=np.float32); vEE[idxE] = eff_EE
                vEI = np.zeros(CFG.N_E, dtype=np.float32); vEI[idxE] = eff_EI
                ring_E_EE[slot] = vEE
                ring_E_EI[slot] = vEI

            # I presyn rings (I→E, I→I)
            ring_I_IE[slot].fill(0.0)
            ring_I_II[slot].fill(0.0)
            I_idx = np.where(spikesI)[0]
            if I_idx.size > 0:
                eff_IE, idxI = stp_IE.on_spikes(t_abs, I_idx)
                eff_II, _    = stp_II.on_spikes(t_abs, I_idx)
                vIE = np.zeros(CFG.N_I, dtype=np.float32); vIE[idxI] = eff_IE
                vII = np.zeros(CFG.N_I, dtype=np.float32); vII[idxI] = eff_II
                ring_I_IE[slot] = vIE
                ring_I_II[slot] = vII

            # progress bar update
            if pbar is not None:
                done += 1
                # bump in ~tick-sized chunks; avoid overshoot at the very end
                if (done % tick) == 0 or (t_abs == T_total - 1):
                    pbar.update(min(tick, T_total - pbar.n))


        # segment boundary snapshot (optional)
        snapshots[t0 + seg["T_ms"]] = (stdp.W_inpE.copy(), stdp.W_EE.copy())
        t0 += seg["T_ms"]
    
    # Prepare outputs
    E_spike_times = [np.array(s, dtype=int) for s in E_spike_times]
    I_spike_times = [np.array(s, dtype=int) for s in I_spike_times]

    if pbar is not None:
        pbar.close()


    
    return dict(
        E_spike_times=E_spike_times,
        I_spike_times=I_spike_times,
        weights=dict(W_inpE=stdp.W_inpE, W_EE=stdp.W_EE),
        init_weights=dict(W_inpE=stdp.init_inpE, W_EE=stdp.init_EE),
        snapshots=snapshots,
        schedule=schedule,
        onsets=onsets,
        patterns=patterns,
	inp_spikes=inp_spikes
    )
